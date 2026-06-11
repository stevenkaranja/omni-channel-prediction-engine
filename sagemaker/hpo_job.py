"""Hyperparameter optimisation job via SageMaker."""
import sagemaker
from sagemaker.tuner import (
    IntegerParameter, ContinuousParameter, HyperparameterTuner
)
from sagemaker.pytorch import PyTorch


def run_hpo(role_arn: str, train_uri: str, val_uri: str) -> str:
    estimator = PyTorch(
        entry_point="train.py",
        role=role_arn,
        instance_type="ml.p3.2xlarge",
        instance_count=1,
        framework_version="2.1.0",
        py_version="py311",
    )

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="val:loss",
        objective_type="Minimize",
        hyperparameter_ranges={
            "lr":         ContinuousParameter(1e-5, 1e-3),
            "d-model":    IntegerParameter(128, 512),
            "nhead":      IntegerParameter(4, 16),
            "num-layers": IntegerParameter(2, 8),
            "batch-size": IntegerParameter(32, 128),
        },
        max_jobs=20,
        max_parallel_jobs=4,
    )

    tuner.fit({"train": train_uri, "val": val_uri}, wait=False)
    print(f"HPO job: {tuner.latest_tuning_job.name}")
    return tuner.latest_tuning_job.name
