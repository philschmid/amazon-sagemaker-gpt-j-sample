import os
import sys
import shutil
import tarfile
import argparse
import boto3
import torch
from transformers import AutoTokenizer, GPTJForCausalLM


def compress(tar_dir=None, output_file="model.tar.gz"):
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(tar_dir, arcname=os.path.sep)


def upload_file_to_s3(bucket_name=None, file_name="model.tar.gz", key_prefix=""):
    s3 = boto3.resource("s3")
    key_prefix_with_file_name = os.path.join(key_prefix, file_name)
    s3.Bucket(bucket_name).upload_file(file_name, key_prefix_with_file_name)
    return f"s3://{bucket_name}/{key_prefix_with_file_name}"


def convert(bucket_name="hf-sagemaker-inference"):
    model_save_dir = "./tmp"
    key_prefix = "gpt-j"
    src_inference_script = os.path.join("code", "inference.py")
    dst_inference_script = os.path.join(model_save_dir, "code")

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(dst_inference_script, exist_ok=True)

    # load fp 16 model
    print("Loading model from `EleutherAI/gpt-j-6b`")
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16
    )
    print("saving model with `torch.save`")
    torch.save(model, os.path.join(model_save_dir, "gptj.pt"))

    print("saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
    tokenizer.save_pretrained(model_save_dir)

    # copy inference script
    print("copying inference.py script")
    shutil.copy(src_inference_script, dst_inference_script)

    # create archive
    print("creating `model.tar.gz` archive")
    compress(model_save_dir)

    # upload to s3
    print(
        f"uploading `model.tar.gz` archive to s3://{bucket_name}/{key_prefix}/model.tar.gz"
    )
    model_uri = upload_file_to_s3(bucket_name=bucket_name, key_prefix=key_prefix)
    print(f"Successfully uploaded to {model_uri}")
    
    sys.stdout.write(model_uri)
    return model_uri


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    # parse args
    args = parse_args()

    if not args.bucket_name:
        raise ValueError(
            "please provide a valid `bucket_name`, when running `python convert_gptj.py --bucket_name` "
        )

    # read config file
    convert(args.bucket_name)
