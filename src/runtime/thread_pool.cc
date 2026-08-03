/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file thread_pool.cc
 * \brief Threadpool for multi-threading runtime.
 */
#include <dmlc/thread_local.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/threading_backend.h>
#if TVM_THREADPOOL_USE_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include "../support/utils.h"
const constexpr int kL1CacheBytes = 64;

namespace tvm {
namespace runtime {
namespace {
using support::IsNumber;

// Route-1 FTZ/DAZ: flush subnormal floats to zero to dodge the 10-100x
// subnormal-arithmetic latency penalty. Set once per thread (worker entry +
// parallel-launch entry) so every compute thread runs flushed. Opt-in via
// TVM_ENABLE_FTZ_DAZ; CHANGES NUMERICS (subnormal -> 0), so never default-on.
// Scope: native pthread pool only (same as the adaptive-threads hook). Under an
// OpenMP build the worker threads bypass RunWorker and are not covered here.
inline void MaybeEnableFtzDaz() {
  static thread_local bool done = false;
  if (done) return;
  done = true;
  static const bool kEnable = []() {
    const char* v = getenv("TVM_ENABLE_FTZ_DAZ");
    return v != nullptr && atoi(v) != 0;
  }();
  if (!kEnable) return;
#if defined(__x86_64__) || defined(_M_X64)
  unsigned int mxcsr = __builtin_ia32_stmxcsr();
  mxcsr |= 0x8040u;  // FTZ (bit15=0x8000) | DAZ (bit6=0x0040)
  __builtin_ia32_ldmxcsr(mxcsr);
#elif defined(__aarch64__)
  uint64_t fpcr;
  __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
  fpcr |= (1ULL << 24);  // FPCR.FZ: flush subnormal inputs and results
  __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
  __asm__ volatile("isb");  // sync so the new FPCR governs subsequent FP ops
#endif
}

constexpr uint32_t kDefaultSpinCount = 300000;

uint32_t GetSpinCount() {
  const char* val = getenv("TVM_THREAD_POOL_SPIN_COUNT");
  if (!val) {
    return kDefaultSpinCount;
  }
  return atoi(val);
}

}  // namespace

// stride in the page, fit to cache line.
constexpr int kSyncStride = 64 / sizeof(std::atomic<int>);

/*!
 * \brief Thread local main environment.
 */
class ParallelLauncher {
 public:
  // Reset the task request.
  void Init(FTVMParallelLambda flambda, void* cdata, int num_task, bool need_sync) {
    num_pending_.store(num_task);
    this->cdata = cdata;
    this->flambda = flambda;
    this->env.num_task = num_task;
    has_error_.store(false);
    // reshape
    if (static_cast<size_t>(num_task) > par_errors_.size()) {
      par_errors_.resize(num_task + 1);
      if (need_sync) {
        delete[] sync_counter_;
        sync_counter_ = new std::atomic<int>[num_task * kSyncStride];
      }
    }
    if (need_sync) {
      for (int i = 0; i < num_task; ++i) {
        sync_counter_[i * kSyncStride].store(0, std::memory_order_relaxed);
      }
      this->env.sync_handle = sync_counter_;
    } else {
      this->env.sync_handle = nullptr;
    }
  }
  ~ParallelLauncher() { delete[] sync_counter_; }
  // Wait n jobs to finish
  int WaitForJobs() {
    while (num_pending_.load() != 0) {
      tvm::runtime::threading::YieldThread();
    }
    if (!has_error_.load()) return 0;
    std::ostringstream os;
    for (size_t i = 0; i < par_errors_.size(); ++i) {
      if (par_errors_[i] != nullptr) {
        if (par_errors_[i]) {
          os << "Task " << i << " error: " << (*par_errors_[i]).what();
        }
        par_errors_[i] = nullptr;
      }
    }
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", os.str().c_str());
    return -1;
  }
  // Signal that one job has finished.
  void SignalJobError(int task_id) {
    num_pending_.fetch_sub(1);
    par_errors_[task_id] = tvm::ffi::details::MoveFromSafeCallRaised();
    has_error_.store(true);
  }
  // Signal that one job has finished.
  void SignalJobFinish() { num_pending_.fetch_sub(1); }
  // Get thread local version of the store.
  static ParallelLauncher* ThreadLocal() { return dmlc::ThreadLocalStore<ParallelLauncher>::Get(); }
  // The parallel lambda
  FTVMParallelLambda flambda;
  // The closure data
  void* cdata;
  // Local env
  TVMParallelGroupEnv env;
  // Whether this thread is worker of the pool.
  // used to prevent recursive launch.
  bool is_worker{false};

 private:
  // The pending jobs.
  std::atomic<int32_t> num_pending_;
  // Whether error has been countered.
  std::atomic<bool> has_error_;
  // The counter page.
  std::atomic<int32_t>* sync_counter_{nullptr};
  // The error message
  std::vector<ffi::Optional<tvm::ffi::Error>> par_errors_;
};

/*! \brief Lock-free single-producer-single-consumer queue for each thread */
class SpscTaskQueue {
 public:
  /*! \brief The task entry */
  struct Task {
    ParallelLauncher* launcher;
    int32_t task_id;
  };

  SpscTaskQueue() : buffer_(new Task[kRingSize]), head_(0), tail_(0) {}

  ~SpscTaskQueue() { delete[] buffer_; }

  /*!
   * \brief Push a task into the queue and notify the comsumer if it is on wait.
   * \param input The task to be dequeued.
   */
  void Push(const Task& input) {
    while (!Enqueue(input)) {
      tvm::runtime::threading::YieldThread();
    }
    if (pending_.fetch_add(1) == -1) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.notify_one();
    }
  }

  /*!
   * \brief Pop a task out of the queue and condition wait if no tasks.
   * \param output The pointer to the task to be dequeued.
   * \param spin_count The number of iterations to spin before sleep.
   * \return Whether pop is successful (true) or we need to exit now (false).
   */
  bool Pop(Task* output, uint32_t spin_count) {
    // Busy wait a bit when the queue is empty.
    // If a new task comes to the queue quickly, this wait avoid the worker from sleeping.
    // The default spin count is set by following the typical omp convention
    for (uint32_t i = 0; i < spin_count && pending_.load() == 0; ++i) {
      tvm::runtime::threading::YieldThread();
    }
    if (pending_.fetch_sub(1) == 0) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return pending_.load() >= 0 || exit_now_.load(); });
    }
    if (exit_now_.load(std::memory_order_relaxed)) {
      return false;
    }
    const uint32_t head = head_.load(std::memory_order_relaxed);
    // sanity check if the queue is empty
    ICHECK(tail_.load(std::memory_order_acquire) != head);
    *output = buffer_[head];
    head_.store((head + 1) % kRingSize, std::memory_order_release);
    return true;
  }

  /*!
   * \brief Signal to terminate the worker.
   */
  void SignalForKill() {
    std::lock_guard<std::mutex> lock(mutex_);
    exit_now_.store(true);
    cv_.notify_all();
  }

 protected:
  /*!
   * \brief Lock-free enqueue.
   * \param input The task to be enqueued.
   * \return Whether the task is enqueued.
   */
  bool Enqueue(const Task& input) {
    if (exit_now_.load(std::memory_order_relaxed)) return false;

    const uint32_t tail = tail_.load(std::memory_order_relaxed);

    if ((tail + 1) % kRingSize != (head_.load(std::memory_order_acquire))) {
      buffer_[tail] = input;
      tail_.store((tail + 1) % kRingSize, std::memory_order_release);
      return true;
    }
    return false;
  }

  // the cache line paddings are used for avoid false sharing between atomic variables
  typedef char cache_line_pad_t[kL1CacheBytes];
  cache_line_pad_t pad0_;
  // size of the queue, the queue can host size_ - 1 items at most
  // define it as a constant for better compiler optimization
  static constexpr const int kRingSize = 2;
  // pointer to access the item
  Task* const buffer_;

  cache_line_pad_t pad1_;
  // queue head, where one gets a task from the queue
  std::atomic<uint32_t> head_;

  cache_line_pad_t pad2_;
  // queue tail, when one puts a task to the queue
  std::atomic<uint32_t> tail_;

  cache_line_pad_t pad3_;
  // pending tasks in the queue
  std::atomic<int8_t> pending_{0};

  cache_line_pad_t pad4_;
  // signal for exit now
  std::atomic<bool> exit_now_{false};

  // internal mutex
  std::mutex mutex_;
  // cv for consumer
  std::condition_variable cv_;
};

// The thread pool
class ThreadPool {
 public:
  ThreadPool() : num_workers_(tvm::runtime::threading::MaxConcurrency()) {
    const char* exclude_worker0 = getenv("TVM_EXCLUDE_WORKER0");
    if (exclude_worker0 && atoi(exclude_worker0) == 0) {
      exclude_worker0_ = false;
    }
    Init();
  }

  ~ThreadPool() {
    for (std::unique_ptr<SpscTaskQueue>& q : queues_) {
      q->SignalForKill();
    }
    threads_.reset();
  }

  void Reset() {
    for (std::unique_ptr<SpscTaskQueue>& q : queues_) {
      q->SignalForKill();
    }
    // Destroy threads before we destory the shared queue, otherwise we segfault on MacOS
    threads_.reset();
    queues_.clear();
    Init();
  }

  int Launch(FTVMParallelLambda flambda, void* cdata, int num_task, int need_sync) {
    ParallelLauncher* launcher = ParallelLauncher::ThreadLocal();
    ICHECK(!launcher->is_worker)
        << "Cannot launch parallel job inside worker, consider fuse then parallel";
    if (num_task == 0) {
      num_task = num_workers_used_;
    }
    if (need_sync != 0) {
      ICHECK_LE(num_task, num_workers_used_)
          << "Request parallel sync task larger than number of threads used "
          << " workers=" << num_workers_used_ << " request=" << num_task;
    }
    launcher->Init(flambda, cdata, num_task, need_sync != 0);
    SpscTaskQueue::Task tsk;
    tsk.launcher = launcher;
    // if worker0 is taken by the main, queues_[0] is abandoned
    for (int i = exclude_worker0_; i < num_task; ++i) {
      tsk.task_id = i;
      queues_[i]->Push(tsk);
    }
    // use the main thread to run task 0
    if (exclude_worker0_) {
      TVMParallelGroupEnv* penv = &(tsk.launcher->env);
      if ((*tsk.launcher->flambda)(0, penv, cdata) == 0) {
        tsk.launcher->SignalJobFinish();
      } else {
        tsk.launcher->SignalJobError(tsk.task_id);
      }
    }
    int res = launcher->WaitForJobs();
    return res;
  }

  static ThreadPool* ThreadLocal() { return dmlc::ThreadLocalStore<ThreadPool>::Get(); }

  void UpdateWorkerConfiguration(threading::ThreadGroup::AffinityMode mode, int nthreads,
                                 const std::vector<unsigned int>& cpus) {
    // this will also reset the affinity of the ThreadGroup
    // may use less than the MaxConcurrency number of workers
    num_workers_used_ = threads_->Configure(mode, nthreads, exclude_worker0_, cpus);
    // if MaxConcurrency restricted the number of workers (e.g., due to
    // hyperthreading), respect the restriction
    num_workers_used_ = std::min(num_workers_, num_workers_used_);
  }

  int32_t NumThreads() const { return num_workers_used_; }

 private:
  // Shared initialization code
  void Init() {
    for (int i = 0; i < num_workers_; ++i) {
      // The SpscTaskQueue only hosts ONE item at a time
      queues_.emplace_back(std::make_unique<SpscTaskQueue>());
    }
    threads_ = std::make_unique<tvm::runtime::threading::ThreadGroup>(
        num_workers_, [this](int worker_id) { this->RunWorker(worker_id); },
        exclude_worker0_ /* include_main_thread */);
    num_workers_used_ = threads_->Configure(threading::ThreadGroup::kBig, 0, exclude_worker0_);
  }

  // Internal worker function.
  void RunWorker(int worker_id) {
    SpscTaskQueue* queue = queues_[worker_id].get();
    SpscTaskQueue::Task task;
    ParallelLauncher::ThreadLocal()->is_worker = true;
    MaybeEnableFtzDaz();  // FTZ/DAZ for this pool worker (one-shot, env-gated)
    // Initialize the spin count (from envvar TVM_THREAD_POOL_SPIN_COUNT) on
    // the global first use of the ThreadPool.
    // TODO(tulloch): should we make this configurable via standard APIs?
    static size_t spin_count = GetSpinCount();
    while (queue->Pop(&task, spin_count)) {
      ICHECK(task.launcher != nullptr);
      TVMParallelGroupEnv* penv = &(task.launcher->env);
      void* cdata = task.launcher->cdata;
      if ((*task.launcher->flambda)(task.task_id, penv, cdata) == 0) {
        task.launcher->SignalJobFinish();
      } else {
        task.launcher->SignalJobError(task.task_id);
      }
    }
  }
  int num_workers_;
  // number of workers used (can be restricted with affinity pref)
  int num_workers_used_;
  // if or not to exclude worker 0 and use main to run task 0
  bool exclude_worker0_{true};
  std::vector<std::unique_ptr<SpscTaskQueue>> queues_;
  std::unique_ptr<tvm::runtime::threading::ThreadGroup> threads_;
};

/*!
 * \brief args[0] is the AffinityMode, args[1] is the number of threads.
 *  args2 is a list of CPUs which is used to set the CPU affinity.
 */
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("runtime.config_threadpool",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    threading::ThreadGroup::AffinityMode mode =
                        static_cast<threading::ThreadGroup::AffinityMode>(args[0].cast<int>());
                    int nthreads = args[1].cast<int>();
                    std::vector<unsigned int> cpus;
                    if (args.size() >= 3) {
                      auto cpu_array = args[2].cast<ffi::Array<ffi::String>>();
                      for (auto cpu : cpu_array) {
                        ICHECK(IsNumber(cpu))
                            << "The CPU core information '" << cpu << "' is not a number.";
                        cpus.push_back(std::stoi(cpu));
                      }
                    }
                    threading::Configure(mode, nthreads, cpus);
                  })
      .def("runtime.NumThreads", []() -> int32_t { return threading::NumThreads(); });
}

namespace threading {

#if TVM_THREADPOOL_USE_OPENMP
/*!
 * \brief Helper function that allows to pin threads to cores in case of multi instance execution
 *        when we use OpenMP thread pool.
 *
 * \param mode Affinity mode (now supports only kSpecifyOneCorePerThread and
 *             kSpecifyThreadShareAllCore).
 * \param nthreads The number of threads to use (0 = use all).
 * \param cpus A list of CPU ids to set 'cpu affinity'.
 *
 */
static void ConfigureOMP(tvm::runtime::threading::ThreadGroup::AffinityMode mode, int nthreads,
                         const std::vector<unsigned int>& cpus) {
#if defined(__linux__) || defined(__ANDROID__)
  const int num_workers = MaxConcurrency();

  if (mode == ThreadGroup::kSpecifyOneCorePerThread) {
#pragma omp parallel num_threads(num_workers)
    {
      int core_id = cpus[omp_get_thread_num()];
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(core_id, &cpuset);
#if defined(__ANDROID__)
      sched_setaffinity(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
    }
  } else if (mode == ThreadGroup::kSpecifyThreadShareAllCore) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto id : cpus) {
      CPU_SET(id, &cpuset);
    }

#pragma omp parallel num_threads(num_workers)
    {
#if defined(__ANDROID__)
      sched_setaffinity(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
    }
  }
#endif
}

#endif

void ResetThreadPool() { tvm::runtime::ThreadPool::ThreadLocal()->Reset(); }
/*!
 * \brief configure the CPU id affinity
 * \param mode The preferred CPU type (1 = big, -1 = little, -2 = kSpecifyOneCorePerThread,
 *  -3 = kSpecifyThreadShareAllCore).
 * \param nthreads The number of threads to use (0 = use all).
 * \param cpus cpus A list of CPUs is used to set the 'cpu affinity' for the worker threads.
 *
 */
TVM_DLL void Configure(tvm::runtime::threading::ThreadGroup::AffinityMode mode, int nthreads,
                       std::vector<unsigned int> cpus) {
  tvm::runtime::threading::SetMaxConcurrency(cpus.size());
#if !TVM_THREADPOOL_USE_OPENMP
  tvm::runtime::ThreadPool::ThreadLocal()->UpdateWorkerConfiguration(mode, nthreads, cpus);
#else
  ConfigureOMP(mode, nthreads, cpus);
#endif
}
int32_t NumThreads() { return tvm::runtime::ThreadPool::ThreadLocal()->NumThreads(); }
}  // namespace threading
}  // namespace runtime
}  // namespace tvm

namespace tvm {
namespace runtime {
namespace {

// ---------------------------------------------------------------------------
// Warmup-based calibrated adaptive per-op thread count (TVM 0.22 CPU migration,
// design section "Phase 2"). This is the UNIVERSAL hook: every CPU parallel
// kernel funnels through TVMBackendParallelLaunch with num_task==0 regardless of
// Relax VM exec_mode (bytecode RunInstrCall path AND compiled __vmtir__main /
// tvm_call_cpacked path both reach here), so calibrating here covers both modes.
// The Relax VM func_pool_ wrapper does NOT cover compiled mode, because compiled
// kernels are invoked via tvm_call_cpacked (direct symbol), bypassing func_pool_.
//
// Default-off: when TVM_ADAPTIVE_THREADS is unset/0 the whole feature compiles
// down to a single cached bool check and the launch path is byte-for-byte
// upstream behavior.
struct AdaptiveConfig {
  bool enabled = false;
  int min_threads = 1;       // TVM_MIN_NUM_THREADS: the "few threads" candidate
  uint64_t warmup = 80;      // TVM_WARMUP_TIMES: calls sampled before deciding
  AdaptiveConfig() {
    const char* e = std::getenv("TVM_ADAPTIVE_THREADS");
    enabled = (e != nullptr && std::atoi(e) == 1);
    const char* m = std::getenv("TVM_MIN_NUM_THREADS");
    if (m != nullptr) min_threads = std::atoi(m);
    if (min_threads < 1) min_threads = 1;
    const char* w = std::getenv("TVM_WARMUP_TIMES");
    if (w != nullptr) warmup = std::strtoull(w, nullptr, 10);
    if (warmup < 2) warmup = 2;
  }
};

// Read env once per process; deployment-wide and stable. Mutable storage so tests can
// drive the calibration deterministically (see SetAdaptiveConfigForTesting); production
// only ever reads it after the one-time env-based construction.
AdaptiveConfig& MutableAdaptiveConfig() {
  static AdaptiveConfig cfg;
  return cfg;
}
const AdaptiveConfig& GetAdaptiveConfig() { return MutableAdaptiveConfig(); }

// One profile per distinct parallel region. The region is keyed by the compiled
// parallel-lambda function pointer (flambda), which is stable per region for the
// process lifetime. EMA of measured launch latency at full vs min threads.
struct ParallelRegionProfile {
  uint64_t calls = 0;
  double ema_full = 0.0;
  double ema_min = 0.0;
  bool logged = false;
};

// thread_local: in the Triton TVM backend each model instance runs on a dedicated
// thread and TVMBackendParallelLaunch is called from that instance thread (worker
// threads execute flambda but never re-enter here). So a thread_local map is
// naturally per-instance and lock-free.
thread_local std::unordered_map<uintptr_t, ParallelRegionProfile> g_region_profiles;

// Chrome Trace timeline of parallel-region launches (the 0.9 custom
// TVM_TIMELINE_ENABLE, ported to this universal hook so it covers BOTH relax VM
// exec_modes — the VM profiler's per-op report is bytecode-only). Every launch
// records wall-clock start + duration + the thread count actually used, so the
// gaps BETWEEN regions and the adaptive per-region decision are both visible.
// Env vocabulary matches 0.9:
//   TVM_TIMELINE_ENABLE=1        turn on (default off: cost is one cached bool)
//   TVM_TIMELINE_OUTPUT_FILE=f   output path (default tvm_timeline.json), written
//                                once at process exit
//   TVM_TIMELINE_MAX_EVENTS=n    recording cap (default 1e6); excess is counted
//                                and reported, never silently dropped
//   TVM_TIMELINE_CAPTURE_REQUEST=N  1-based: only record the Nth `main` inference
//                                (directed capture). Default unset = full mode
//                                (record every request for the whole process).
// Scope: native pthread pool, num_workers>1 launches only (the single-worker
// inline path and OpenMP builds are not recorded).
struct TimelineConfig {
  bool enabled = false;
  std::string path = "tvm_timeline.json";
  uint64_t max_events = 1000000;
  int capture_request = -1;  // >0 = directed (Nth main only); -1 = full mode
  TimelineConfig() {
    const char* e = std::getenv("TVM_TIMELINE_ENABLE");
    enabled = (e != nullptr && std::atoi(e) == 1);
    const char* p = std::getenv("TVM_TIMELINE_OUTPUT_FILE");
    if (p != nullptr && p[0] != '\0') path = p;
    const char* m = std::getenv("TVM_TIMELINE_MAX_EVENTS");
    if (m != nullptr) max_events = std::strtoull(m, nullptr, 10);
    const char* cr = std::getenv("TVM_TIMELINE_CAPTURE_REQUEST");
    if (cr != nullptr && cr[0] != '\0') {
      char* endp = nullptr;
      errno = 0;
      long v = std::strtol(cr, &endp, 10);
      // FR-3: LLP64 (Windows MSVC) — long is 32-bit, strtol saturates at
      // LONG_MAX == INT_MAX and sets errno=ERANGE on overflow. Without errno
      // check, "99999999999" would pass v <= 0x7fffffff.
      if (endp != cr && *endp == '\0' && errno != ERANGE &&
          v > 0 && v <= std::numeric_limits<int>::max()) {
        capture_request = static_cast<int>(v);
      } else {
        std::cerr << "[TIMELINE] invalid TVM_TIMELINE_CAPTURE_REQUEST='" << cr
                  << "' (must be positive integer), falling back to full mode" << std::endl;
      }
    }
  }
};
const TimelineConfig& GetTimelineConfig() {
  static TimelineConfig cfg;
  return cfg;
}

struct TimelineEvent {
  uintptr_t region;   // parallel-launch track: flambda key; kernel track: 0
  int32_t name_idx;   // kernel track: index into named_; parallel-launch track: -1
  uint64_t tid;
  double ts_us;
  double dur_us;
  int num_task;  // kernel track: -1
  int8_t phase;  // 0 = steady/adaptive-off, 1 = warmup FULL sample, 2 = warmup MIN sample
  const char* cat;  // static string: "parallel_region" | "kernel" | "inference"
  int request_index = -1;  // >= 0 for cat=inference root events; -1 otherwise
};

// Per-thread capture context: avoids cross-VM interference when two VM instances
// run on different threads simultaneously. Set by EnterCapture on the VM's main
// thread; Record/RecordNamed read it on the same thread (parallel-launch Record
// runs on the launching thread, kernel Begin/End on the executing thread — both
// are the VM's main thread for relax compiled/bytecode execution).
thread_local bool tl_capture_active = false;
thread_local int tl_capture_req_index = -1;
thread_local std::chrono::steady_clock::time_point tl_capture_start{};

class TimelineRecorder {
 public:
  // Region naming: parallel lambdas are LOCAL symbols in the model .so, invisible
  // to dladdr (dynamic symtab only). dladdr still gives module path + load base,
  // so we emit module+offset per event and symbolize_timeline.py maps offsets to
  // the local `<kernel>_compute_` symbols via `nm -n` offline. If dladdr does
  // resolve a name (exported symbol), it is used directly.
  struct RegionInfo {
    std::string name;
    std::string module;
    uintptr_t offset = 0;
  };

  static TimelineRecorder& Global() {
    static TimelineRecorder inst;  // destructor at static teardown writes the file
    return inst;
  }

  void Record(const void* region, std::chrono::steady_clock::time_point t0,
              std::chrono::steady_clock::time_point t1, int num_task, int phase) {
    // Directed-capture gate: in full mode (capture_request < 0) pass through; in
    // directed mode (capture_request > 0) only record while inside the target request.
    if (GetTimelineConfig().capture_request > 0 && !tl_capture_active) return;
    // tid = launching thread: one trace row per model-instance thread.
    uint64_t tid = std::hash<std::thread::id>()(std::this_thread::get_id()) & 0xfffff;
    uintptr_t key = reinterpret_cast<uintptr_t>(region);
    std::lock_guard<std::mutex> lock(mu_);
    if (!epoch_valid_) {
      epoch_ = t0;  // first event defines t=0 (Global() is constructed after its t0)
      epoch_valid_ = true;
    }
    if (regions_.find(key) == regions_.end()) {
      // Resolve NOW, not at dump: the dump runs at static teardown, after the model
      // .so may have been dlclosed, when dladdr can no longer find the image.
      RegionInfo ri;
#if !defined(_WIN32)
      Dl_info info;
      if (dladdr(region, &info) != 0) {
        if (info.dli_sname != nullptr) ri.name = info.dli_sname;
        if (info.dli_fname != nullptr) {
          ri.module = info.dli_fname;
          ri.offset = key - reinterpret_cast<uintptr_t>(info.dli_fbase);
        }
      }
#endif
      if (ri.name.empty()) {
        std::ostringstream oss;
        oss << "region_0x" << std::hex << key;
        ri.name = oss.str();
      }
      regions_.emplace(key, std::move(ri));
    }
    if (events_.size() >= GetTimelineConfig().max_events) {
      ++dropped_;
      return;
    }
    events_.push_back({key, -1, tid, std::chrono::duration<double, std::micro>(t0 - epoch_).count(),
                       std::chrono::duration<double, std::micro>(t1 - t0).count(), num_task,
                       static_cast<int8_t>(phase), "parallel_region"});
    dirty_ = true;
  }

  // Kernel-track event from TVMBackendTimelineBegin/End instrumentation. The name is a
  // string constant inside the (possibly later-dlclosed) model .so, so it is interned
  // into named_ immediately — same lifetime lesson as the dladdr-at-Record rule above.
  void RecordNamed(const char* name, std::chrono::steady_clock::time_point t0,
                   std::chrono::steady_clock::time_point t1) {
    if (GetTimelineConfig().capture_request > 0 && !tl_capture_active) return;
    uint64_t tid = std::hash<std::thread::id>()(std::this_thread::get_id()) & 0xfffff;
    std::lock_guard<std::mutex> lock(mu_);
    if (!epoch_valid_) {
      epoch_ = t0;
      epoch_valid_ = true;
    }
    auto it = name_ids_.find(name);
    if (it == name_ids_.end()) {
      it = name_ids_.emplace(name, static_cast<int32_t>(named_.size())).first;
      named_.emplace_back(name);
    }
    if (events_.size() >= GetTimelineConfig().max_events) {
      ++dropped_;
      return;
    }
    events_.push_back({0, it->second, tid,
                       std::chrono::duration<double, std::micro>(t0 - epoch_).count(),
                       std::chrono::duration<double, std::micro>(t1 - t0).count(), -1, 0,
                        "kernel"});
    dirty_ = true;
  }

  // Directed-capture API: called by the VM layer around the Nth `main` invocation.
  // EnterCapture marks the recorder as "inside target request" so Record/RecordNamed
  // stop early-returning; ExitCapture pushes a cat=inference root event spanning the
  // captured request and clears the flag. In full mode (capture_request < 0) the VM
  // calls these for every request so the trace always has per-request boundaries.
  void EnterCapture(int req_index) {
    tl_capture_active = true;
    tl_capture_req_index = req_index;
    tl_capture_start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mu_);
    if (!epoch_valid_) {
      epoch_ = tl_capture_start;
      epoch_valid_ = true;
    }
  }

  void ExitCapture() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!tl_capture_active) return;
    auto now = std::chrono::steady_clock::now();
    uint64_t tid = std::hash<std::thread::id>()(std::this_thread::get_id()) & 0xfffff;
    if (events_.size() < GetTimelineConfig().max_events) {
      events_.push_back({0, -1, tid,
                         std::chrono::duration<double, std::micro>(tl_capture_start - epoch_).count(),
                         std::chrono::duration<double, std::micro>(now - tl_capture_start).count(),
                         -1, 0, "inference", tl_capture_req_index});
      dirty_ = true;
    } else {
      ++dropped_;
    }
    tl_capture_active = false;
    // Directed-capture immediate publish: the target request just completed, so all
    // its kernel / parallel_region / inference events are already in events_. Long-
    // running hosts (Tomcat/JNI/FFM) never reach static teardown — publish now so the
    // trace is visible without stopping the service. WriteTraceLocked runs under mu_
    // (already held here), publishes a snapshot without mutating events_, and on
    // success clears dirty_ so the destructor does not redundantly re-write.
    if (GetTimelineConfig().capture_request > 0) {
      WriteTraceLocked();
    }
  }

  ~TimelineRecorder() {
    std::lock_guard<std::mutex> lock(mu_);
    if (!dirty_) {
      // Already published (events_ on disk, no appends since last WriteTraceLocked) or
      // nothing ever happened. Warn only when directed capture was configured but the
      // target request was never reached (events_ still empty at teardown).
      if (events_.empty() && GetTimelineConfig().capture_request > 0) {
        std::cerr << "[TIMELINE] TVM_TIMELINE_CAPTURE_REQUEST="
                  << GetTimelineConfig().capture_request
                  << " but target request was not reached; no trace written" << std::endl;
      }
      return;
    }
    // dirty: unwritten appends exist (full mode throughout, or a directed flush that
    // failed and left events_ unpublished). Final publish at static teardown.
    WriteTraceLocked();
  }

 private:
  // Publish events_ as a Chrome-Trace JSON to cfg.path. MUST be called under mu_.
  // Returns true on success (clears dirty_), false on failure (keeps dirty_ so the
  // destructor retries at static teardown).
  //
  // Snapshot semantics (砚砚 review #2): the canonical events_ is NEVER mutated —
  // rebase + stable_sort run on a local copy, so events_ stays on its original epoch
  // and later appends + re-flushes (multiple VMs each hitting their target N) keep one
  // consistent timeline. The old destructor mutated events_ in place, which corrupts a
  // second flush (rebased-old + raw-new timestamps coexist).
  //
  // Atomic publish (砚砚 review #3): write <path>.partial, flush+verify, then rename
  // onto <path>. An external observer polling the path (Tomcat host) never sees a
  // half-written JSON; it sees either the previous complete trace or the new one.
  bool WriteTraceLocked() {
    if (events_.empty()) {
      dirty_ = false;
      return true;
    }
    const TimelineConfig& cfg = GetTimelineConfig();
    std::vector<TimelineEvent> snap = events_;  // copy — canonical events_ untouched
    // Negative-ts fix on the snapshot only: the epoch is set by whichever event first
    // reaches Record, which can be an inner parallel_region that finished before the
    // outer kernel that started earlier — leaving the outer event with ts < 0 (Perfetto
    // drops them). Rebase all events to the earliest start time and stable-sort.
    double min_ts = snap[0].ts_us;
    for (const auto& ev : snap) {
      if (ev.ts_us < min_ts) min_ts = ev.ts_us;
    }
    if (min_ts < 0.0) {
      for (auto& ev : snap) ev.ts_us -= min_ts;
    }
    std::stable_sort(snap.begin(), snap.end(),
                     [](const TimelineEvent& a, const TimelineEvent& b) {
                       return a.ts_us < b.ts_us;
                     });
    // Atomic publish: temp file in the same directory (same filesystem → POSIX rename
    // is atomic). Windows std::rename fails if the target exists; acceptable for this
    // Linux/macOS-targeted debugging tool.
    std::string tmp_path = cfg.path + ".partial";
    std::ofstream os(tmp_path);
    if (!os) {
      std::cerr << "[TIMELINE] cannot open " << tmp_path << std::endl;
      return false;
    }
    os << std::fixed << std::setprecision(3);
    os << "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[";
    static const char* kPhase[] = {"steady", "warmup_full", "warmup_min"};
    bool first = true;
    const RegionInfo kUnknown;  // defensive only: Record always registers the region
    for (const TimelineEvent& ev : snap) {
      if (!first) os << ",";
      first = false;
      if (ev.request_index >= 0) {  // cat=inference root event spanning one main call
        os << "\n{\"name\":\"inference\",\"ph\":\"X\",\"cat\":\"" << ev.cat
           << "\",\"pid\":1,\"tid\":" << ev.tid << ",\"ts\":" << ev.ts_us
           << ",\"dur\":" << ev.dur_us << ",\"args\":{\"request_index\":" << ev.request_index
           << "}}";
        continue;
      }
      if (ev.name_idx >= 0) {  // kernel track (Begin/End instrumentation)
        os << "\n{\"name\":\"" << named_[ev.name_idx] << "\",\"ph\":\"X\",\"cat\":\"" << ev.cat
           << "\",\"pid\":1,\"tid\":" << ev.tid << ",\"ts\":" << ev.ts_us
           << ",\"dur\":" << ev.dur_us << "}";
        continue;
      }
      auto it = regions_.find(ev.region);
      const RegionInfo& ri = (it != regions_.end()) ? it->second : kUnknown;
      os << "\n{\"name\":\"" << ri.name << "\",\"ph\":\"X\",\"cat\":\"" << ev.cat
         << "\",\"pid\":1,\"tid\":" << ev.tid << ",\"ts\":" << ev.ts_us << ",\"dur\":" << ev.dur_us
         << ",\"args\":{\"num_task\":" << ev.num_task << ",\"phase\":\"" << kPhase[ev.phase]
         << "\",\"module\":\"" << ri.module << "\",\"offset\":\"0x" << std::hex << ri.offset
         << std::dec << "\"}}";
    }
    os << "\n]}\n";
    os.flush();  // force write; check state before declaring success
    if (!os) {
      // std::cerr, not LOG(INFO): at static teardown the logging infra may be gone.
      std::cerr << "[TIMELINE] write failed to " << tmp_path << std::endl;
      os.close();
      std::remove(tmp_path.c_str());
      return false;
    }
    os.close();
    if (std::rename(tmp_path.c_str(), cfg.path.c_str()) != 0) {
      std::cerr << "[TIMELINE] cannot rename " << tmp_path << " -> " << cfg.path << std::endl;
      std::remove(tmp_path.c_str());
      return false;
    }
    std::cerr << "[TIMELINE] wrote " << snap.size() << " events to " << cfg.path;
    if (dropped_ > 0) std::cerr << " (dropped " << dropped_ << " over TVM_TIMELINE_MAX_EVENTS)";
    std::cerr << std::endl;
    dirty_ = false;
    return true;
  }

  std::mutex mu_;
  std::vector<TimelineEvent> events_;
  std::unordered_map<uintptr_t, RegionInfo> regions_;
  // Kernel-track name interning. Keyed by the string-constant pointer (stable per
  // loaded module, one map hit per call site) with the text copied into named_.
  std::unordered_map<const char*, int32_t> name_ids_;
  std::vector<std::string> named_;
  std::chrono::steady_clock::time_point epoch_;
  bool epoch_valid_ = false;
  uint64_t dropped_ = 0;
  // True when events_ has appends since the last successful WriteTraceLocked. Set on
  // every Record/RecordNamed/ExitCapture push; cleared on a successful publish. The
  // destructor uses this (not a one-shot "done" bool) so that: (a) a failed early flush
  // is retried at teardown, and (b) in multi-VM directed mode, VM-B's appends after
  // VM-A's flush are still published (砚砚 review #1 — a global done flag would skip).
  bool dirty_ = false;
};

// Per-thread stack of open TVMBackendTimelineBegin frames. Kernels are entered and
// left on the same (launching) thread; a stack keeps nested/fused calls correct.
thread_local std::vector<std::pair<const char*, std::chrono::steady_clock::time_point>>
    g_timeline_kernel_stack;

}  // namespace

// Test-only hooks: the production config is read from env exactly once and cannot be
// toggled per-test (it caches), and the profile map is thread_local. These let the C++
// unit test (tests/cpp/threading_backend_test.cc) drive the calibration deterministically.
// Not declared in any public header; not for production use.
// TODO(merge-gate): for the production release build these should be guarded behind a
// TVM_ENABLE_TESTING build flag (libtvm built without it for release, with it for the
// test build) so they don't land in the shipped ABI. Left exported here because this
// experimental build links cpptest directly against libtvm, which needs the symbols.
TVM_DLL void SetAdaptiveConfigForTesting(bool enabled, int min_threads, uint64_t warmup) {
  AdaptiveConfig& cfg = MutableAdaptiveConfig();
  cfg.enabled = enabled;
  cfg.min_threads = (min_threads < 1) ? 1 : min_threads;
  cfg.warmup = (warmup < 2) ? 2 : warmup;
}
TVM_DLL void ResetAdaptiveProfilesForTesting() { g_region_profiles.clear(); }

}  // namespace runtime
}  // namespace tvm

int TVMBackendParallelLaunch(FTVMParallelLambda flambda, void* cdata, int num_task) {
  tvm::runtime::MaybeEnableFtzDaz();  // FTZ/DAZ for the launching thread (runs task 0)
  int num_workers = tvm::runtime::threading::MaxConcurrency();
  // Per-op thread-local override (TVM 0.22 CPU runtime migration design 6.1):
  // only override when the caller did not request an explicit task count
  // (num_task == 0), so existing callers with a fixed num_task are untouched.
  // Clamp to the LIVE pool's NumThreads(), NOT MaxConcurrency(): the two diverge when
  // threading::Configure() shrank the pool (explicit nthreads, or Phase 3 affinity), and
  // an override larger than the live pool would otherwise abort in ThreadPool::Launch
  // (ICHECK_LE). Same fix as the adaptive path below.
  int override_num_task = tvm::runtime::threading::GetPerOpNumThreads();
  if (override_num_task > 0 && num_task == 0) {
    int avail = tvm::runtime::threading::NumThreads();
    int cap = avail >= 1 ? avail : num_workers;
    num_task = std::min(override_num_task, cap);
  }

  // Calibrated adaptive per-region thread selection. Only engages when the caller
  // left num_task unspecified (==0, the codegen convention) and no manual override
  // is active, so explicit/overridden callers keep precedence. num_workers==1 has
  // nothing to tune.
  //
  // Gated to the native pthread pool: the calibration is a timing/EMA state machine
  // driven by the ThreadPool::Launch sampling below, which the OpenMP launch path does
  // NOT run. Under an OpenMP build (TVM_THREADPOOL_USE_OPENMP=1) adaptive is simply
  // absent — the launch falls through to plain full-thread OpenMP (= upstream behavior).
#if !TVM_THREADPOOL_USE_OPENMP
  const auto& acfg = tvm::runtime::GetAdaptiveConfig();
  int sampling = 0;  // 0 = apply decision, 1 = sampling full, 2 = sampling min
  tvm::runtime::ParallelRegionProfile* prof = nullptr;
  if (acfg.enabled && num_task == 0 && num_workers > 1) {
    prof = &tvm::runtime::g_region_profiles[reinterpret_cast<uintptr_t>(flambda)];
    if (prof->calls < acfg.warmup) {
      // Interleave FULL/MIN across the warmup window (even call -> FULL, odd -> MIN) so
      // neither candidate is systematically measured colder than the other. A block
      // "first-half FULL, second-half MIN" would sample MIN in the warmer second half and
      // bias the decision toward MIN.
      if ((prof->calls & 1) == 0) {
        num_task = 0;                 // sample at full threads (resolved to all)
        sampling = 1;
      } else {
        num_task = acfg.min_threads;  // sample at min threads
        sampling = 2;
      }
    } else {
      // Calibrated: pick min only if it actually measured faster than full.
      // SCOPE: the decision is made once and kept for the process lifetime. This is
      // correct for static-shape compiled TIR regions (the validated target: constant
      // batch size, same loop extents every call). It is NOT designed for regions whose
      // cost varies at runtime (dynamic shape, or a shared runtime-helper lambda reused
      // across very different sizes): a region first seen small could stay pinned to MIN.
      // We don't re-sample because (a) the deployment is static-bs and (b) the workload
      // extent is not visible at this layer (it is inside the opaque `cdata` closure).
      num_task = (prof->ema_min > 0.0 && prof->ema_min < prof->ema_full) ? acfg.min_threads : 0;
      if (!prof->logged) {
        LOG(INFO) << "[ADAPTIVE] region=" << reinterpret_cast<void*>(flambda)
                  << " full_us=" << prof->ema_full << " min_us=" << prof->ema_min
                  << (num_task == 0 ? " => FULL" : " => MIN");
        prof->logged = true;
      }
    }
    // Clamp to the LIVE ThreadLocal pool size -- this is what ThreadPool::Launch
    // ICHECK_LE's against. Do NOT clamp to MaxConcurrency(): the two diverge when
    // threading::Configure() shrank the pool (an explicit nthreads, or a future
    // per-instance affinity slice), and a TVM_MIN_NUM_THREADS larger than the live
    // pool would otherwise abort inside Launch.
    int avail = tvm::runtime::threading::NumThreads();
    if (avail >= 1 && num_task > avail) num_task = avail;
  }
#endif  // !TVM_THREADPOOL_USE_OPENMP

  if (num_workers == 1) {
    std::atomic<int32_t> sync_counter{0};
    TVMParallelGroupEnv env;
    env.num_task = 1;
    env.sync_handle = &sync_counter;
    (*flambda)(0, &env, cdata);
    return 0;
  } else {
#if !TVM_THREADPOOL_USE_OPENMP
    if (sampling != 0) {
      auto t0 = std::chrono::steady_clock::now();
      int res = tvm::runtime::ThreadPool::ThreadLocal()->Launch(flambda, cdata, num_task, 1);
      auto t1 = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      if (sampling == 1) {
        prof->ema_full = (prof->ema_full == 0.0) ? us : 0.5 * prof->ema_full + 0.5 * us;
      } else {
        prof->ema_min = (prof->ema_min == 0.0) ? us : 0.5 * prof->ema_min + 0.5 * us;
      }
      prof->calls++;
      if (tvm::runtime::GetTimelineConfig().enabled) {
        int eff = num_task == 0 ? tvm::runtime::threading::NumThreads() : num_task;
        tvm::runtime::TimelineRecorder::Global().Record(reinterpret_cast<void*>(flambda), t0, t1,
                                                        eff, sampling);
      }
      return res;
    }
    if (tvm::runtime::GetTimelineConfig().enabled) {
      auto t0 = std::chrono::steady_clock::now();
      int res = tvm::runtime::ThreadPool::ThreadLocal()->Launch(flambda, cdata, num_task, 1);
      auto t1 = std::chrono::steady_clock::now();
      int eff = num_task == 0 ? tvm::runtime::threading::NumThreads() : num_task;
      tvm::runtime::TimelineRecorder::Global().Record(reinterpret_cast<void*>(flambda), t0, t1,
                                                      eff, /*phase=steady*/ 0);
      return res;
    }
    int res = tvm::runtime::ThreadPool::ThreadLocal()->Launch(flambda, cdata, num_task, 1);
    return res;
#else
    if (num_task == 0) num_task = num_workers;
    omp_set_num_threads(num_task);
#pragma omp parallel num_threads(num_task)
    {
      TVMParallelGroupEnv env;
      env.num_task = num_task;
      (*flambda)(omp_get_thread_num(), &env, cdata);
    }
    return 0;
#endif
  }
}

// Kernel-track instrumentation entry points (fork extension, see c_backend_api.h).
// Emitted as call_extern at PrimFunc entry/exit by the compile-side
// instrument_timeline pass; resolved from libtvm at .so load exactly like
// TVMBackendParallelLaunch, so they work in BOTH relax VM exec_modes and cover
// kernels that have no parallel loop (which never reach the launch hook above).
int TVMBackendTimelineBegin(const char* name) {
  if (!tvm::runtime::GetTimelineConfig().enabled) return 0;
  tvm::runtime::g_timeline_kernel_stack.emplace_back(name, std::chrono::steady_clock::now());
  return 0;
}

int TVMBackendTimelineEnd(void) {
  if (!tvm::runtime::GetTimelineConfig().enabled) return 0;
  auto& stack = tvm::runtime::g_timeline_kernel_stack;
  if (stack.empty()) return 0;  // unmatched End (e.g. env flipped mid-run): ignore
  auto frame = stack.back();
  stack.pop_back();
  tvm::runtime::TimelineRecorder::Global().RecordNamed(frame.first, frame.second,
                                                       std::chrono::steady_clock::now());
  return 0;
}

// VM-layer directed-capture control (called from vm.cc around main invocation).
// Thin wrappers so vm.cc does not need to reach into thread_pool.cc's anonymous
// namespace; same pattern as TVMBackendTimelineBegin/End above.
int TVMTimelineEnterCapture(int req_index) {
  if (!tvm::runtime::GetTimelineConfig().enabled) return 0;
  tvm::runtime::TimelineRecorder::Global().EnterCapture(req_index);
  return 0;
}

int TVMTimelineExitCapture() {
  if (!tvm::runtime::GetTimelineConfig().enabled) return 0;
  tvm::runtime::TimelineRecorder::Global().ExitCapture();
  return 0;
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv) {
#if TVM_THREADPOOL_USE_OPENMP
#pragma omp barrier
#else
  using tvm::runtime::kSyncStride;
  int num_task = penv->num_task;
  std::atomic<int>* sync_counter = reinterpret_cast<std::atomic<int>*>(penv->sync_handle);
  int old_counter = sync_counter[task_id * kSyncStride].fetch_add(1, std::memory_order_release);
  for (int i = 0; i < num_task; ++i) {
    if (i != task_id) {
      while (sync_counter[i * kSyncStride].load(std::memory_order_relaxed) <= old_counter) {
        tvm::runtime::threading::YieldThread();
      }
    }
  }
  std::atomic_thread_fence(std::memory_order_acquire);
#endif
  return 0;
}
