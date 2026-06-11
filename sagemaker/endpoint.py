"""Deploy trained model to SageMaker managed endpoint."""
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel


def deploy_endpoint(model_artifact: str, role_arn: str, endpoint_name: str) -> str:
    model = PyTorchModel(
        model_data=model_artifact,
        role=role_arn,
        framework_version="2.1.0",
        py_version="py311",
        entry_point="inference.py",
    )

    predictor = model.deploy(
        initial_instance_count=2,
        instance_type="ml.g4dn.xlarge",
        endpoint_name=endpoint_name,
        data_capture_config=sagemaker.model_monitor.DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=20,
            destination_s3_uri="s3://omni-channel-models/data-capture/",
        ),
    )
    print(f"Endpoint live: {predictor.endpoint_name}")
    return predictor.endpoint_name


def predict_batch(endpoint_name: str, payload: dict) -> dict:
    client = boto3.client("sagemaker-runtime")
    import json
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read())
