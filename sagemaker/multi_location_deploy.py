"""Deploy forecast endpoints for all 400 retail locations."""
import boto3
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed


REGIONS = ["eu-west-1", "us-east-1", "ap-southeast-1"]


def deploy_region(region: str, model_artifact: str, role_arn: str) -> str:
    import sagemaker
    from sagemaker.pytorch import PyTorchModel
    session = sagemaker.Session(boto3.Session(region_name=region))
    model = PyTorchModel(
        model_data=model_artifact, role=role_arn,
        framework_version="2.1.0", py_version="py311",
        entry_point="inference.py", sagemaker_session=session,
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g4dn.xlarge",
        endpoint_name=f"omni-channel-forecast-{region}",
    )
    return predictor.endpoint_name


def deploy_all(model_artifact: str, role_arn: str):
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(deploy_region, r, model_artifact, role_arn): r for r in REGIONS}
        for f in as_completed(futures):
            print(f"✓ {futures[f]}: {f.result()}")
