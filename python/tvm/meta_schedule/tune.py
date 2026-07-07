# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The core tuning API"""
import logging
import os
from typing import List, Optional

from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .measure_callback import MeasureCallback
from .runner import Runner
from .task_scheduler import TaskScheduler
from .tune_context import TuneContext
from .post_optimization import PostOpt

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Cross-session cost-model reuse (fork extension, default-inert):
# when TVM_MS_COST_MODEL is set and the caller asked for the default "xgb"
# cost model by name, load the model from that path before tuning (if the
# file exists) and save it back after tuning completes successfully.
_COST_MODEL_ENV = "TVM_MS_COST_MODEL"


def _cost_model_reuse_path(cost_model_spec) -> Optional[str]:
    path = os.environ.get(_COST_MODEL_ENV, "").strip()
    if not path:
        return None
    if cost_model_spec != "xgb":
        logger.warning(
            "[ms-cost-model] %s is set but cost_model=%r is not 'xgb'; ignoring it",
            _COST_MODEL_ENV,
            cost_model_spec,
        )
        return None
    return path


def _load_cost_model(path: str, num_cores: int) -> Optional[CostModel]:
    try:
        cost_model = CostModel.create("xgb", num_tuning_cores=num_cores, tree_method="auto")
        cost_model.load(path)
    except Exception as err:  # pylint: disable=broad-except
        logger.warning(
            "[ms-cost-model] failed to load cost model from %s (%s: %s); "
            "falling back to a fresh model",
            path,
            type(err).__name__,
            err,
        )
        return None
    logger.warning(
        "[ms-cost-model] loaded cost model from %s (data_size=%s)",
        path,
        getattr(cost_model, "data_size", "n/a"),
    )
    return cost_model


def _save_cost_model(cost_model: CostModel, path: str) -> None:
    # Atomic write: a crash mid-save must never clobber a healthy model file.
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        cost_model.save(tmp_path)
        os.replace(tmp_path, path)
    except Exception as err:  # pylint: disable=broad-except
        logger.warning(
            "[ms-cost-model] failed to save cost model to %s (%s: %s)",
            path,
            type(err).__name__,
            err,
        )
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return
    logger.warning(
        "[ms-cost-model] saved cost model to %s (data_size=%s)",
        path,
        getattr(cost_model, "data_size", "n/a"),
    )


def tune_tasks(
    *,
    tasks: List[TuneContext],
    task_weights: List[float],
    work_dir: str,
    max_trials_global: int,
    max_trials_per_task: Optional[int] = None,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "gradient",
    module_equality: str = "structural",
    post_optimization: Optional[bool] = False,
) -> Database:
    """Tune a list of tasks. Using a task scheduler.

    Parameters
    ----------
    tasks : List[TuneContext]
        The list of tasks to tune.
    task_weights : List[float]
        The weight of each task.
    work_dir : str
        The working directory.
    max_trials_global : int
        The maximum number of trials to run globally.
    max_trials_per_task : Optional[int]
        The maximum number of trials to run per task.
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : Builder.BuilderType
        The builder.
    runner : Runner.RunnerType
        The runner.
    database : Database.DatabaseType
        The database.
    cost_model : CostModel.CostModelType
        The cost model.
    measure_callbacks : MeasureCallback.CallbackListType
        The measure callbacks.
    task_scheduler : TaskScheduler.TaskSchedulerType
        The task scheduler.
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:

            - "structural": Use StructuralEqual/Hash
            - "ignore-tensor": Same as "structural", but ignore tensor raw data during equality
                testing and hashing.
            - "anchor-block": Apply equality testing and hashing on the anchor block extracted from
                a given module. The "ignore-tensor" varint is used for the extracted blocks or in
                case no anchor block is found. For the definition of the anchor block, see
                tir/analysis/analysis.py.
    post_optimization : Optional[Bool]
        Generate post-optimization using Droplet Search as exploitation space.

    Returns
    -------
    database : Database
        The database with all tuning records
    """
    if len(tasks) == 0:
        raise ValueError("No tasks to tune.")

    if len(tasks) != len(task_weights):
        raise ValueError(
            f"Length of tasks ({len(tasks)}) and task_weights ({len(task_weights)}) do not match."
        )

    num_cores = tasks[0].num_threads

    if max_trials_per_task is None:
        max_trials_per_task = max_trials_global
    if not isinstance(builder, Builder):
        builder = Builder.create(builder, max_workers=num_cores)
    if not isinstance(runner, Runner):
        runner = Runner.create(runner, max_workers=num_cores)
    if database == "json":
        database = Database.create(database, work_dir=work_dir, module_equality=module_equality)
    elif not isinstance(database, Database):
        database = Database.create(database, module_equality=module_equality)
    cost_model_reuse_path = None
    if not isinstance(cost_model, CostModel):
        cost_model_reuse_path = _cost_model_reuse_path(cost_model)
        loaded = None
        if cost_model_reuse_path and os.path.exists(cost_model_reuse_path):
            loaded = _load_cost_model(cost_model_reuse_path, num_cores)
        if loaded is not None:
            cost_model = loaded
        else:
            cost_model = CostModel.create(cost_model, num_tuning_cores=num_cores, tree_method="auto")
    if isinstance(measure_callbacks, MeasureCallback):
        measure_callbacks = [measure_callbacks]
    elif measure_callbacks == "default":
        measure_callbacks = MeasureCallback.create(measure_callbacks)
    if not isinstance(task_scheduler, TaskScheduler):
        task_scheduler = TaskScheduler.create(task_scheduler)
    task_scheduler.tune(
        tasks=tasks,
        task_weights=task_weights,
        max_trials_global=max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        measure_callbacks=measure_callbacks,
        database=database,
        cost_model=cost_model,
    )
    # Save only after tuning completed successfully; a tuning exception must
    # propagate without overwriting the previous model file.
    if cost_model_reuse_path is not None:
        _save_cost_model(cost_model, cost_model_reuse_path)
    if post_optimization:
        post_opt = PostOpt(work_dir, tasks[0].target)
        post_opt.run()
    return database
