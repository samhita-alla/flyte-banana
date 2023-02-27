import base64
import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import evaluate
import flytekit
import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from flytekit import Resources, Secret, approve, task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.types.structured.structured_dataset import StructuredDataset
from huggingface_hub import model_info
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

load_dotenv()

HUGGINGFACE_REPO = "fine-tuned-bert"
SECRET_GROUP = "deployment-secrets"
SECRET_NAME = "flyte-banana-creds"
HF_SECRET_GROUP = "hf-secrets"
HF_SECRET_NAME = "flyte-banana-hf-creds"
datasets_tuple = NamedTuple(
    "datasets", train_dataset=StructuredDataset, eval_dataset=StructuredDataset
)


def create_local_dir(dir_name: str):
    working_dir = flytekit.current_context().working_directory
    local_dir = Path(working_dir) / dir_name
    local_dir.mkdir(exist_ok=True)
    return local_dir


@task(
    cache=True,
    cache_version="1.0",
    requests=Resources(mem="1Gi", cpu="2", ephemeral_storage="500Mi"),
)
def download_dataset() -> FlyteDirectory:
    dataset = load_dataset("yelp_review_full")

    local_dir = create_local_dir(dir_name="yelp_data")

    dataset.save_to_disk(dataset_dict_path=local_dir)
    return FlyteDirectory(path=str(local_dir))


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


@task(
    cache=True,
    cache_version="1.0",
    requests=Resources(mem="1Gi", cpu="2", ephemeral_storage="500Mi"),
)
def tokenize(dataset: FlyteDirectory) -> FlyteDirectory:
    downloaded_path = dataset.download()
    loaded_dataset = load_from_disk(downloaded_path)
    tokenized_dataset = loaded_dataset.map(tokenize_function, batched=True)

    local_dir = create_local_dir(dir_name="yelp_tokenized_data")

    tokenized_dataset.save_to_disk(dataset_dict_path=local_dir)
    return FlyteDirectory(path=str(local_dir))


if os.getenv("DEMO") != "":
    mem = "1Gi"
    gpu = "0"
    dataset_size = 10
else:
    # mem = "1Gi"
    # gpu = "0"
    # dataset_size = 10
    mem = "3Gi"
    gpu = "1"
    dataset_size = 100


@task(requests=Resources(mem="1Gi", cpu="2"))
def get_train_eval(
    tokenized_dataset: FlyteDirectory,
) -> datasets_tuple:
    downloaded_path = tokenized_dataset.download()
    loaded_tokenized_dataset = load_from_disk(downloaded_path)

    small_train_dataset = (
        loaded_tokenized_dataset["train"].shuffle(seed=42).select(range(dataset_size))
    )
    small_eval_dataset = (
        loaded_tokenized_dataset["test"].shuffle(seed=42).select(range(dataset_size))
    )
    return datasets_tuple(
        train_dataset=StructuredDataset(dataframe=small_train_dataset),
        eval_dataset=StructuredDataset(dataframe=small_eval_dataset),
    )


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


@task(
    requests=Resources(mem=mem, cpu="2", gpu=gpu),
    secret_requests=[
        Secret(
            group=HF_SECRET_GROUP,
            key=HF_SECRET_NAME,
            mount_requirement=Secret.MountType.FILE,
        )
    ],
)
def train(
    small_train_df: StructuredDataset, small_eval_df: StructuredDataset, hf_user: str
) -> dict:
    HUGGING_FACE_HUB_TOKEN = flytekit.current_context().secrets.get(
        SECRET_GROUP, HF_SECRET_NAME
    )
    repo = f"{hf_user}/{HUGGINGFACE_REPO}"
    execution_id = flytekit.current_context().execution_id.name
    small_train_dataset = small_train_df.open(Dataset).all()
    small_eval_dataset = small_eval_df.open(Dataset).all()

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=5
    )

    training_args = TrainingArguments(
        output_dir=HUGGINGFACE_REPO,
        evaluation_strategy="epoch",
        push_to_hub=True,
        hub_token=HUGGING_FACE_HUB_TOKEN,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.push_to_hub(
        commit_message=f"End of training - Flyte execution ID {execution_id}"
    )
    return {
        "sha": model_info(repo).sha,
        "execution_id": execution_id,
        "repo": repo,
    }


@task(
    secret_requests=[
        Secret(
            group=SECRET_GROUP, key=SECRET_NAME, mount_requirement=Secret.MountType.FILE
        )
    ]
)
def push_to_github(
    model_metadata: dict, gh_owner: str, gh_repo: str, gh_branch: str
) -> str:
    token = flytekit.current_context().secrets.get(SECRET_GROUP, SECRET_NAME)

    additions = [
        {
            "path": f"banana/model_metadata.json",
            "contents": f"{base64.urlsafe_b64encode(json.dumps(model_metadata).encode()).decode('utf-8')}",
        }
    ]

    sha_result = subprocess.run(
        f"""
        curl -s -H "Authorization: bearer {token}" \
        -H "Accept: application/vnd.github.VERSION.sha" \
        "https://api.github.com/repos/{gh_owner}/{gh_repo}/commits/main"
        """,
        shell=True,
        capture_output=True,
        text=True,
    )

    github_sha = sha_result.stdout

    cmd = f"""
curl https://api.github.com/graphql -s -H "Authorization: bearer {token}" --data @- << GRAPHQL | jq '.data.createCommitOnBranch.commit.url[0:56]'
{{
    "query": "mutation (\$input: CreateCommitOnBranchInput!) {{
    createCommitOnBranch(input: \$input) {{
        commit {{
          url 
        }} 
    }} 
    }}",
    "variables": {{
      "input": {{
        "branch": {{
          "repositoryNameWithOwner": "{gh_owner}/{gh_repo}",
          "branchName": "{gh_branch}"
        }},
        "message": {{
        "headline": "Update the model artifact"
        }},
        "fileChanges": {{
          "additions": {json.dumps(additions)}
        }},
        "expectedHeadOid": "{github_sha}"
      }}
    }}
}}
GRAPHQL
"""

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
    )

    return result.stdout


@workflow
def yelp_pipeline(gh_owner: str, gh_repo: str, gh_branch: str, hf_user: str) -> str:
    dataset = download_dataset()
    tokenized_dataset = tokenize(dataset=dataset)
    split_dataset = get_train_eval(tokenized_dataset=tokenized_dataset)
    model_metadata = train(
        small_eval_df=split_dataset.train_dataset,
        small_train_df=split_dataset.eval_dataset,
        hf_user=hf_user,
    )
    approve_to_deploy = approve(
        model_metadata, "deploy-model", timeout=timedelta(hours=1)
    )
    commit_model = push_to_github(
        model_metadata=model_metadata,
        gh_owner=gh_owner,
        gh_repo=gh_repo,
        gh_branch=gh_branch,
    )
    approve_to_deploy >> commit_model

    return commit_model


if __name__ == "__main__":
    print(
        yelp_pipeline(
            gh_owner="samhita-alla",
            gh_repo="flyte-banana",
            gh_branch="main",
            hf_user="Samhita",
        )
    )
