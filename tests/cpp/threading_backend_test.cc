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

#include <gtest/gtest.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/threading_backend.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

constexpr size_t N = 128;
void AtomicCompute(int task_id, size_t n, std::atomic<size_t>* acc, TVMParallelGroupEnv* penv) {
  const size_t N_per_task = (n + penv->num_task - 1) / penv->num_task;
  for (size_t i = task_id * N_per_task; i < n && i < (task_id + 1) * N_per_task; ++i) {
    acc->fetch_add(i, std::memory_order_relaxed);
  }
  return;
}

class AffinityCheck {
 public:
  AffinityCheck(uint32_t parent_id, int max_concurrency, std::atomic<size_t>* acc)
      : id_(parent_id), max_concurrency_(max_concurrency), acc_(acc) {}

  void Compute(int task_id, size_t n, TVMParallelGroupEnv* penv) {
    AtomicCompute(task_id, n, acc_, penv);
  }

  int GetComputeResult() { return acc_->load(std::memory_order_relaxed); }

  void GetAffinity(int task_id) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    std::lock_guard<std::mutex> lock(mutex_);
    thread_affinity_[task_id] = cpuset;
    // Printing the current thread CPU affinity.
    std::ostringstream str;
    for (int i = 0; i < max_concurrency_; i++) {
      if (CPU_ISSET(i, &cpuset)) {
        str << i << ",";
      }
    }
#endif
  }

  bool VerifyAffinity(const std::vector<uint32_t>& cpus) {
#if defined(__linux__)
    std::unordered_set<uint32_t> uset;
    cpu_set_t cpu_mask;
    CPU_ZERO(&cpu_mask);
    for (auto x : cpus) {
      CPU_SET(x, &cpu_mask);
      uset.insert(x);
    }

    for (auto x : thread_affinity_) {
      if (!CPU_EQUAL(&cpu_mask, &x.second)) {
        bool cpu_find = false;
        for (auto cpu : uset) {
          CPU_ISSET(cpu, &x.second);
          uset.erase(cpu);
          cpu_find = true;
          break;
        }
        if (!cpu_find) return false;
      }
    }
#endif
    return true;
  }

 private:
  uint32_t id_;
  int max_concurrency_;
  std::atomic<size_t>* acc_;
  std::mutex mutex_;
#if defined(__linux__)
  std::unordered_map<int, cpu_set_t> thread_affinity_;
#endif
};

static FTVMParallelLambda atomic_add_task_id = [](int task_id, TVMParallelGroupEnv* penv,
                                                  void* cdata) -> int {
  auto* data = reinterpret_cast<std::atomic<size_t>*>(cdata);
  AtomicCompute(task_id, N, data, penv);
  return 0;
};

static FTVMParallelLambda affinity_check_task_id = [](int task_id, TVMParallelGroupEnv* penv,
                                                      void* cdata) -> int {
  auto* data = reinterpret_cast<AffinityCheck*>(cdata);
  data->Compute(task_id, N, penv);
  data->GetAffinity(task_id);
  return 0;
};

TEST(ThreadingBackend, TVMBackendParallelLaunch) {
  std::atomic<size_t> acc(0);
  TVMBackendParallelLaunch(atomic_add_task_id, &acc, 0);
  EXPECT_EQ(acc.load(std::memory_order_relaxed), N * (N - 1) / 2);
}

TEST(ThreadingBackend, TVMBackendParallelLaunchMultipleThreads) {
  // TODO(tulloch) use parameterised tests when available.
  size_t num_jobs_per_thread = 3;
  size_t max_num_threads = 2;

  for (size_t num_threads = 1; num_threads < max_num_threads; ++num_threads) {
    std::vector<std::unique_ptr<std::thread>> ts;
    for (size_t i = 0; i < num_threads; ++i) {
      ts.emplace_back(new std::thread([&]() {
        for (size_t j = 0; j < num_jobs_per_thread; ++j) {
          std::atomic<size_t> acc(0);
          TVMBackendParallelLaunch(atomic_add_task_id, &acc, 0);
          EXPECT_EQ(acc.load(std::memory_order_relaxed), N * (N - 1) / 2);
        }
      }));
    }
    for (auto& t : ts) {
      t->join();
    }
  }
}

TEST(ThreadingBackend, TVMBackendAffinityConfigure) {
  int max_concurrency = tvm::runtime::threading::MaxConcurrency();
  std::vector<std::unique_ptr<std::thread>> ts;
  // Returning as there is only one CPU available.
  if (max_concurrency <= 1) {
    return;
  }
  // Creating two threads to test the 'CPU list affinity' feature.
  const int threads_num = 2;
  // Getting the maximum number of CPUs which are available to each thread.
  const int cpus_num_per_thread = max_concurrency / threads_num;
  // Testing two mode of affinity.,
  std::vector<tvm::runtime::threading::ThreadGroup::AffinityMode> modes = {
      tvm::runtime::threading::ThreadGroup::kSpecifyOneCorePerThread,
      tvm::runtime::threading::ThreadGroup::kSpecifyThreadShareAllCore};
  for (auto mode : modes) {
    for (int thread_pool_idx = 0; thread_pool_idx < threads_num; thread_pool_idx++) {
      ts.emplace_back(new std::thread(
          [&](int thread_pool_index, int sys_max_concurrency,
              tvm::runtime::threading::ThreadGroup::AffinityMode affinity_mode) {
            std::atomic<size_t> acc(0);
            AffinityCheck ac(thread_pool_index, sys_max_concurrency, &acc);
            std::vector<unsigned int> cpus;
            for (int k = 0; k < cpus_num_per_thread; k++) {
              cpus.push_back(thread_pool_index * cpus_num_per_thread + k);
            }
            tvm::runtime::threading ::Configure(affinity_mode, 0, cpus);
            TVMBackendParallelLaunch(affinity_check_task_id, &ac, 0);
            EXPECT_EQ(ac.GetComputeResult(), N * (N - 1) / 2);
            EXPECT_EQ(ac.VerifyAffinity(cpus), true);
          },
          thread_pool_idx, max_concurrency, mode));
    }
  }
  for (auto& t : ts) {
    t->join();
  }
}

TEST(ThreadingBackend, TVMBackendParallelForWithThreadingBackend) {
  int n = 100;
  std::vector<int> vec(/*size=*/n, /*value=*/0);
  tvm::runtime::parallel_for_with_threading_backend([&vec](int i) { vec[i] = i; }, 0, n);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(vec[i], i);
  }
}

// Phase 1 step 1: TIR-level failing test
// Goal: verify per-op thread-local override can change TVMBackendParallelLaunch num_task
//
// Design doc section 6.1 defines: SetPerOpNumThreads / GetPerOpNumThreads / ClearPerOpNumThreads
//
// THIS TEST WILL NOT COMPILE until Phase 1 step 2 implements the override API.
// Compilation failure is the correct red signal for step 1.
TEST(ThreadingBackend, ThreadLocalNumTaskOverride) {
  const int override_nthreads = 2;

  std::vector<int32_t> observed_num_tasks;
  std::mutex mu;

  auto num_task_tracker = [](int task_id, TVMParallelGroupEnv* penv, void* cdata) -> int {
    auto* ctx = reinterpret_cast<std::pair<std::vector<int32_t>*, std::mutex*>*>(cdata);
    std::lock_guard<std::mutex> lock(*ctx->second);
    ctx->first->push_back(penv->num_task);
    return 0;
  };

  std::pair<std::vector<int32_t>*, std::mutex*> tracker_ctx{&observed_num_tasks, &mu};

  // Case 1: Baseline - explicit num_task parameter is respected
  const int max_concurrency = tvm::runtime::threading::MaxConcurrency();
  if (max_concurrency < 2) {
    GTEST_SKIP() << "Need at least 2 workers for this test";
  }
  const int baseline_num_task = std::min(4, max_concurrency);
  observed_num_tasks.clear();
  TVMBackendParallelLaunch(num_task_tracker, &tracker_ctx, baseline_num_task);
  EXPECT_EQ(observed_num_tasks.size(), baseline_num_task)
      << "num_task=" << baseline_num_task << " should result in " << baseline_num_task << " workers";
  for (int32_t nt : observed_num_tasks) {
    EXPECT_EQ(nt, baseline_num_task) << "each worker should see num_task=" << baseline_num_task;
  }

  // Case 2: Per-op thread-local override (KEY FAILING TEST - compilation error until step 2)
  // The three API calls below use design-doc names from section 6.1.
  // They do not exist yet in v0.22.0, so this test fails to compile.
  // This compilation failure DRIVES step 2: implement SetPerOpNumThreads / GetPerOp / ClearPerOp.
  observed_num_tasks.clear();
  tvm::runtime::threading::SetPerOpNumThreads(override_nthreads);
  TVMBackendParallelLaunch(num_task_tracker, &tracker_ctx, 0);
  EXPECT_EQ(observed_num_tasks.size(), override_nthreads)
      << "override should limit workers to " << override_nthreads;
  for (int32_t nt : observed_num_tasks) {
    EXPECT_EQ(nt, override_nthreads) << "each worker should see override num_task=" << override_nthreads;
  }
  tvm::runtime::threading::ClearPerOpNumThreads();
}

// Phase 2: calibrated adaptive per-op thread count.
// The adaptive logic lives in TVMBackendParallelLaunch (thread_pool.cc), keyed by the
// parallel-lambda pointer. These test-only hooks (defined in thread_pool.cc) let us drive
// the calibration deterministically instead of via the process-cached env vars.
namespace tvm {
namespace runtime {
TVM_DLL void SetAdaptiveConfigForTesting(bool enabled, int min_threads, uint64_t warmup);
TVM_DLL void ResetAdaptiveProfilesForTesting();
}  // namespace runtime
}  // namespace tvm

TEST(ThreadingBackend, AdaptiveThreadsCalibration) {
  const int max_concurrency = tvm::runtime::threading::MaxConcurrency();
  if (max_concurrency < 2) {
    GTEST_SKIP() << "Need at least 2 workers for adaptive calibration";
  }
  const int min_threads = 1;
  const uint64_t warmup = 20;

  // Record the num_task each launch actually runs at: one entry per launch (task_id==0).
  std::vector<int32_t> observed;
  std::mutex mu;
  auto recorder = [](int task_id, TVMParallelGroupEnv* penv, void* cdata) -> int {
    if (task_id == 0) {
      auto* ctx = reinterpret_cast<std::pair<std::vector<int32_t>*, std::mutex*>*>(cdata);
      std::lock_guard<std::mutex> lock(*ctx->second);
      ctx->first->push_back(penv->num_task);
    }
    return 0;  // empty work => dispatch/sync overhead dominates => MIN should win the calibration
  };
  std::pair<std::vector<int32_t>*, std::mutex*> ctx{&observed, &mu};

  // ---- adaptive ON: warmup state machine + calibrated decision ----
  tvm::runtime::SetAdaptiveConfigForTesting(/*enabled=*/true, min_threads, warmup);
  tvm::runtime::ResetAdaptiveProfilesForTesting();
  observed.clear();
  const int total = static_cast<int>(warmup) + 10;
  for (int i = 0; i < total; ++i) {
    TVMBackendParallelLaunch(recorder, &ctx, 0);
  }
  ASSERT_EQ(observed.size(), static_cast<size_t>(total));

  // warmup phase 1 (first half): sampled at FULL threads (the pool's worker count, > min).
  const int32_t full_val = observed[0];
  EXPECT_GT(full_val, min_threads) << "full sampling should use more than min threads";
  for (uint64_t i = 0; i < warmup / 2; ++i) {
    EXPECT_EQ(observed[i], full_val) << "warmup phase 1 call " << i << " should sample at full";
  }
  // warmup phase 2 (second half): sampled at MIN threads.
  for (uint64_t i = warmup / 2; i < warmup; ++i) {
    EXPECT_EQ(observed[i], min_threads) << "warmup phase 2 call " << i << " should sample at min";
  }
  // post-warmup: stable calibrated decision; empty work is dispatch-bound => MIN.
  for (int i = static_cast<int>(warmup); i < total; ++i) {
    EXPECT_EQ(observed[i], min_threads)
        << "post-warmup call " << i << " should pick MIN for empty (dispatch-bound) work";
  }

  // ---- adaptive OFF: upstream behavior, always full ----
  tvm::runtime::SetAdaptiveConfigForTesting(/*enabled=*/false, min_threads, warmup);
  tvm::runtime::ResetAdaptiveProfilesForTesting();
  observed.clear();
  for (int i = 0; i < 5; ++i) {
    TVMBackendParallelLaunch(recorder, &ctx, 0);
  }
  for (int32_t nt : observed) {
    EXPECT_EQ(nt, full_val) << "adaptive OFF must use full threads (upstream behavior)";
  }

  // ---- explicit num_task takes precedence over adaptive (only meaningful with room) ----
  if (full_val >= 3) {
    tvm::runtime::SetAdaptiveConfigForTesting(/*enabled=*/true, min_threads, warmup);
    tvm::runtime::ResetAdaptiveProfilesForTesting();
    observed.clear();
    const int explicit_task = 2;  // distinct from both min(1) and full(>=3)
    for (int i = 0; i < 5; ++i) {
      TVMBackendParallelLaunch(recorder, &ctx, explicit_task);
    }
    for (int32_t nt : observed) {
      EXPECT_EQ(nt, explicit_task) << "explicit num_task must be respected even when adaptive is on";
    }
  }

  // restore default-off so other tests in this binary are unaffected.
  tvm::runtime::SetAdaptiveConfigForTesting(/*enabled=*/false, 1, 80);
  tvm::runtime::ResetAdaptiveProfilesForTesting();
}

// Counterpart to AdaptiveThreadsCalibration: a deterministically compute-bound region
// (each task does total/num_task real work) is ~num_workers x faster at FULL than MIN,
// so the calibration must keep it at FULL. Guards the "don't strangle parallel work"
// direction (the regression 0.9's duration-threshold heuristic caused).
TEST(ThreadingBackend, AdaptiveThreadsPicksFullForComputeBound) {
  const int max_concurrency = tvm::runtime::threading::MaxConcurrency();
  if (max_concurrency < 2) {
    GTEST_SKIP() << "Need at least 2 workers";
  }
  const uint64_t warmup = 8;

  std::vector<int32_t> observed;
  std::mutex mu;
  auto heavy = [](int task_id, TVMParallelGroupEnv* penv, void* cdata) -> int {
    // Real, splittable work: more workers => less per-worker work => faster wall clock.
    const int64_t total = 2000000;
    const int nt = penv->num_task < 1 ? 1 : penv->num_task;
    const int64_t per = (total + nt - 1) / nt;
    const int64_t begin = static_cast<int64_t>(task_id) * per;
    const int64_t end = std::min<int64_t>(total, begin + per);
    volatile double acc = 0.0;
    for (int64_t i = begin; i < end; ++i) acc += static_cast<double>(i) * 1.000001;
    (void)acc;
    if (task_id == 0) {
      auto* ctx = reinterpret_cast<std::pair<std::vector<int32_t>*, std::mutex*>*>(cdata);
      std::lock_guard<std::mutex> lock(*ctx->second);
      ctx->first->push_back(penv->num_task);
    }
    return 0;
  };
  std::pair<std::vector<int32_t>*, std::mutex*> ctx{&observed, &mu};

  tvm::runtime::SetAdaptiveConfigForTesting(/*enabled=*/true, /*min_threads=*/1, warmup);
  tvm::runtime::ResetAdaptiveProfilesForTesting();
  observed.clear();
  const int total = static_cast<int>(warmup) + 6;
  for (int i = 0; i < total; ++i) {
    TVMBackendParallelLaunch(heavy, &ctx, 0);
  }
  ASSERT_EQ(observed.size(), static_cast<size_t>(total));
  const int32_t full_val = observed[0];
  EXPECT_GT(full_val, 1) << "full sampling should use more than 1 thread";
  for (int i = static_cast<int>(warmup); i < total; ++i) {
    EXPECT_EQ(observed[i], full_val)
        << "post-warmup call " << i << " should pick FULL for compute-bound work";
  }

  tvm::runtime::SetAdaptiveConfigForTesting(/*enabled=*/false, 1, 80);
  tvm::runtime::ResetAdaptiveProfilesForTesting();
}

// Regression: a TVM_MIN_NUM_THREADS larger than the live ThreadLocal pool must be clamped
// to the pool's actual NumThreads(), not MaxConcurrency(), and not passed through to
// ThreadPool::Launch (which ICHECK_LE-aborts on num_task > num_workers).
//
// To exercise the case the reviewer raised -- a live pool SMALLER than MaxConcurrency() --
// we run on a dedicated thread and shrink that thread's ThreadLocal pool via
// threading::Configure(..., nthreads<cpus.size(), full_cpu_list): this keeps MaxConcurrency()
// large (== cpus.size()) while NumThreads() drops to `shrunk`. min_threads is set above the
// shrunk pool, so a MaxConcurrency()-based clamp would still overshoot and abort; only the
// NumThreads()-based clamp keeps num_task within the live pool. Using a separate thread keeps
// the resized pool out of the main thread so other tests are unaffected.
TEST(ThreadingBackend, AdaptiveThreadsClampsMinToLivePool) {
  const int max_concurrency = tvm::runtime::threading::MaxConcurrency();
  if (max_concurrency < 4) {
    GTEST_SKIP() << "Need >=4 workers to meaningfully shrink the live pool below MaxConcurrency";
  }
  const uint64_t warmup = 6;
  const int shrunk = 2;                        // live pool size < MaxConcurrency()
  const int oversized_min = max_concurrency;   // > shrunk live pool

  std::vector<int32_t> observed;
  std::mutex mu;
  int32_t pool_live = 0;
  int32_t maxconc_live = 0;
  auto recorder = [](int task_id, TVMParallelGroupEnv* penv, void* cdata) -> int {
    if (task_id == 0) {
      auto* ctx = reinterpret_cast<std::pair<std::vector<int32_t>*, std::mutex*>*>(cdata);
      std::lock_guard<std::mutex> lock(*ctx->second);
      ctx->first->push_back(penv->num_task);
    }
    return 0;
  };
  std::pair<std::vector<int32_t>*, std::mutex*> ctx{&observed, &mu};

  tvm::runtime::SetAdaptiveConfigForTesting(/*enabled=*/true, oversized_min, warmup);

  std::thread worker([&]() {
    // full cpu list (size == MaxConcurrency) but override nthreads=shrunk => pool shrinks,
    // MaxConcurrency stays large => NumThreads() < MaxConcurrency() (the divergence case).
    std::vector<unsigned int> cpus;
    for (int i = 0; i < max_concurrency; ++i) cpus.push_back(i);
    tvm::runtime::threading::Configure(
        tvm::runtime::threading::ThreadGroup::kSpecifyThreadShareAllCore, shrunk, cpus);
    pool_live = tvm::runtime::threading::NumThreads();
    maxconc_live = tvm::runtime::threading::MaxConcurrency();
    tvm::runtime::ResetAdaptiveProfilesForTesting();
    for (int i = 0; i < static_cast<int>(warmup) + 2; ++i) {
      TVMBackendParallelLaunch(recorder, &ctx, 0);  // would abort if clamped to MaxConcurrency
    }
  });
  worker.join();

  EXPECT_EQ(pool_live, shrunk) << "live pool should be shrunk below MaxConcurrency for this test";
  EXPECT_GT(maxconc_live, pool_live) << "test only meaningful when MaxConcurrency > live pool";
  ASSERT_FALSE(observed.empty());
  for (int32_t nt : observed) {
    EXPECT_GE(nt, 1);
    EXPECT_LE(nt, pool_live)
        << "adaptive num_task must be clamped to the LIVE pool (NumThreads), not MaxConcurrency";
  }

  tvm::runtime::SetAdaptiveConfigForTesting(/*enabled=*/false, 1, 80);
}
