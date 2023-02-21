import base64
import json
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

load_dotenv()


SECRET_GROUP = "deployment-secrets"
SECRET_NAME = "flyte-banana-creds"
datasets_tuple = NamedTuple("datasets", train_dataset=Dataset, eval_dataset=Dataset)


def create_local_dir(dir_name: str):
    working_dir = flytekit.current_context().working_directory
    local_dir = Path(working_dir) / dir_name
    local_dir.mkdir(exist_ok=True)
    return local_dir


@task(cache=True, cache_version="1.0", requests=Resources(mem="1Gi", cpu="2"))
def download_dataset() -> FlyteDirectory:
    dataset = load_dataset("yelp_review_full")

    local_dir = create_local_dir(dir_name="yelp_data")

    dataset.save_to_disk(dataset_dict_path=local_dir)
    return FlyteDirectory(path=str(local_dir))


def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


@task(cache=True, cache_version="1.0", requests=Resources(mem="1Gi", cpu="2"))
def tokenize(dataset: FlyteDirectory) -> FlyteDirectory:
    downloaded_path = dataset.download()
    loaded_dataset = load_from_disk(downloaded_path)
    tokenized_datasets = loaded_dataset.map(tokenize_function, batched=True)

    local_dir = create_local_dir(dir_name="yelp_tokenized_data")

    tokenized_datasets.save_to_disk(dataset_dict_path=local_dir)
    return FlyteDirectory(path=str(local_dir))


@task(requests=Resources(mem="1Gi", cpu="2"))
def get_train_eval(
    tokenized_datasets: FlyteDirectory,
) -> datasets_tuple:
    downloaded_path = tokenized_datasets.download()
    loaded_tokenized_datasets = load_from_disk(downloaded_path)

    small_train_dataset = (
        loaded_tokenized_datasets["train"].shuffle(seed=42).select(range(10))
    )
    small_eval_dataset = (
        loaded_tokenized_datasets["test"].shuffle(seed=42).select(range(10))
    )
    return datasets_tuple(
        train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
    )


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


@task(requests=Resources(mem="1Gi", gpu="1"))
def train(small_train_dataset: Dataset, small_eval_dataset: Dataset) -> FlyteDirectory:
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=5
    )

    local_dir = create_local_dir(dir_name="test_trainer")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(local_dir)

    return FlyteDirectory(path=str(local_dir))


@task(secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)])
def push_to_github(
    model_dir: FlyteDirectory, owner: str, repo: str, branch: str
) -> str:
    token = flytekit.current_context().secrets.get(SECRET_GROUP, SECRET_NAME)
    remote_src = model_dir.remote_source or ""
    downloaded_path = model_dir.download()

    files = [f for f in Path(downloaded_path).iterdir() if f.is_file()]
    additions = []

    for each_file in files:
        remote_file_name = str(Path(remote_src) / each_file.name)
        additions.append(
            {
                "path": f"banana/model/{each_file.name}",
                "contents": f"{base64.b64encode(remote_file_name.encode('utf-8')).decode('utf-8')}",
            }
        )

    sha_result = subprocess.run(
        f"""
        curl -s -H "Authorization: bearer {token}" \
        -H "Accept: application/vnd.github.VERSION.sha" \
        "https://api.github.com/repos/{owner}/{repo}/commits/main"
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
          "repositoryNameWithOwner": "{owner}/{repo}",
          "branchName": "{branch}"
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
def yelp_pipeline(owner: str, repo: str, branch: str) -> str:
    dataset = download_dataset()
    tokenized_datasets = tokenize(dataset=dataset)
    split_dataset = get_train_eval(tokenized_datasets=tokenized_datasets)
    model_dir = train(
        small_train_dataset=split_dataset.train_dataset,
        small_eval_dataset=split_dataset.eval_dataset,
    )
    approve_to_deploy = approve(model_dir, "deploy-model", timeout=timedelta(hours=1))
    commit_model = push_to_github(
        model_dir=model_dir, owner=owner, repo=repo, branch=branch
    )
    approve_to_deploy >> commit_model

    return commit_model


if __name__ == "__main__":
    print(yelp_pipeline(owner="samhita-alla", repo="flyte-banana", branch="main"))
