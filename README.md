# sagemaker-gpt-j

This repository contains instruction and code on how to run `GPT-J` for inference using Amazon SageMaker.


## Getting Started

Create `model.tar.gz`. ´compress` & `upload_file_to_s3` methods can be found in the [notebook](notebook-sample.ipynb)

```python
from sagemaker.huggingface import HuggingFaceModel
import sagemaker

# IAM role with permissions to create endpoint
role = sagemaker.get_execution_role()

# public S3 URI to gpt-j artifact
model_uri="s3://huggingface-sagemaker-models/transformers/4.12.3/pytorch/1.9.1/gpt-j/model.tar.gz"

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	model_data=model_uri,
	transformers_version='4.12.3',
	pytorch_version='1.9.1',
	py_version='py38',
	role=role, 
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='ml.g4dn.xlarge' #'ml.p3.2xlarge' # ec2 instance type
)
```