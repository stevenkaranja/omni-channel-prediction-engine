"""Launch SageMaker training job."""
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import yaml, os


def launch_training(config_path: str = "config.yaml") -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    sess = sagemaker.Session()
    estimator = PyTorch(
        entry_point="train.py",
        source_dir=".",
        role=cfg["sagemaker"]["role_arn"],
        instance_type=cfg["sagemaker"]["instance_type"],
        instance_count=cfg["sagemaker"]["instance_count"],
        framework_version="2.1.0",
        py_version="py311",
        output_path=cfg["sagemaker"]["output_path"],
        hyperparameters={
            "d-model": cfg["model"]["d_model"],
            "nhead": cfg["model"]["nhead"],
            "num-layers": cfg["model"]["num_encoder_layers"],
            "batch-size": cfg["training"]["batch_size"],
            "max-epochs": cfg["training"]["max_epochs"],
            "lr": cfg["training"]["learning_rate"],
        },
        metric_definitions=[
            {"Name": "train:loss", "Regex": "train_loss=([0-9.]+)"},
            {"Name": "val:loss",   "Regex": "val_loss=([0-9.]+)"},
        ],
    )

    estimator.fit(
        inputs={"train": cfg["sagemaker"]["train_data_uri"],
                "val":   cfg["sagemaker"]["val_data_uri"]},
        wait=False,
    )
    print(f"Training job: {estimator.latest_training_job.name}")
    return estimator.latest_training_job.name


if __name__ == "__main__":
    launch_training()
