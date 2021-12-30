# sagemaker-gpt-j

This repository contains instruction and code on how to run `GPT-J` for inference using Amazon SageMaker.


## Getting Started

Create `model.tar.gz`. ´compress` & `upload_file_to_s3` methods can be found in the [notebook](notebook-sample.ipynb)

```python
from transformers import AutoTokenizer,GPTJForCausalLM
import torch
import os
import shutil 


model_save_dir="./tmp"
bucket_name="hf-sagemaker-inference"
key_prefix="gpt-j"
src_inference_script= os.path.join("code","inference.py")
dst_inference_script= os.path.join(model_save_dir,"code")

os.makedirs(model_save_dir,exist_ok=True)
os.makedirs(dst_inference_script,exist_ok=True)

# load fp 16 model
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16)
torch.save(model, os.path.join(model_save_dir,"gptj.pt"))

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer.save_pretrained(model_save_dir)

# copy inference script
shutil.copy(src_inference_script, dst_inference_script)

# create archive
compress(model_save_dir)

# upload to s3
model_uri = upload_file_to_s3(bucket_name=bucket_name,key_prefix=key_prefix)
model_uri


```

Deploy endpoint

```python
from sagemaker.huggingface import HuggingFaceModel
import boto3
import os

os.environ["AWS_DEFAULT_REGION"]="us-east-1"


iam_role="sagemaker_execution_role"
model_uri="s3://hf-sagemaker-inference/gpt-j/model.tar.gz"

iam_client = boto3.client('iam')
role = iam_client.get_role(RoleName=iam_role)['Role']['Arn']

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
  model_data=model_uri,
	transformers_version='4.12',
	pytorch_version='1.9',
	py_version='py38',
	role=role, 
)


# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='ml.g4dn.xlarge' #'ml.p3.2xlarge' # ec2 instance type
)


predictor.predict({
	'inputs': "Can you please let us know more details about your "
})
```