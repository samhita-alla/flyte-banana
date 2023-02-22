# Model Development and Inference with Flyte and Banana

A fine-tuned version of [bert-base-cased](https://huggingface.co/bert-base-cased) model orchestrated with Flyte and deployed on Banana.

## 🥞 Stack

The technology stack comprises Flyte and Banana.

### Flyte
[Flyte](https://flyte.org/) is a workflow orchestration platform that allows you to build production-grade data and ML pipelines.
The code for Flyte is located in the [flyte](flyte/) directory.

### Banana
[Banana](https://www.banana.dev/) provides inference hosting for ML models on serverless GPUs.
The code for Banana is located in the [banana](banana/) directory.

## 🛠️ Setup

1. Make a copy of this repository by forking it.
2. Configure your Banana account by creating a deployment and linking to the forked GitHub repo. This can be done by navigating to Team > Integrations > GitHub > Manage Repos.
3. Create a virtual environment and install the required Flyte and Banana dependencies.
4. Obtain access tokens for HuggingFace and GitHub and store them in files within a secrets directory, as follows:

```
<your-secrets-dir>/deployments-secrets/flyte-banana-creds
<your-secrets-dir>/hf-secrets/flyte-banana-hf-creds
```

5. Within a .env file, set the following two variables to enable local code execution:

```bash
FLYTE_SECRETS_DEFAULT_DIR=<your-secrets-dir>
DEMO="1"
```

6. The `flyte/workflows/ml_pipelines.py` file contains tasks and workflows. To verify that everything is functioning properly, run the file locally as follows:

```
pyflyte run ml_pipelines.py --gh_owner <your-github-username> --gh_repo <your-github-repo> --gh_branch <your-github-repo-branch> --hf_user <your-huggingface-username>
```

7. To test if you can properly deploy on Banana, you can also run the Banana server locally. Navigate to the [banana](banana/) directory and execute the command `python server.py`. Then, confirm that the Banana API endpoint is functional by running the [`banana/test.py`](banana/test.py) file.
8. To run the Flyte workflow on an actual Flyte backend, [set up a Flyte cluster](https://docs.flyte.org/en/latest/deployment/deployment/index.html). The simplest way to get started is by running `flytectl demo start` command, which spins up a mini-replica of the Flyte deployment.
9. Register tasks and workflows using the following command, which can leverage the [docker registry included with the demo cluster](https://docs.flyte.org/en/latest/deployment/deployment/sandbox.html#start-the-sandbox) for image pushing and pulling:

```
pyflyte register --image <flyte-docker-image> ml_pipeline.py
```
   
10. Launch the registered workflow on the UI.
11. To deploy the retrained model on Banana, select "Yes".
12. This action saves the model metadata, such as the latest commit SHA, Flyte execution ID, and HuggingFace username, in the corresponding GitHub repository.
13. The GitHub push operation then triggers the Banana deployment process.
14. Copy your Banana API and model keys, and store them in `.env` file.
15. Once the deployment is complete, confirm that the Banana API endpoint is operational by executing [`banana/test_banana_api.py`](banana/test_banana_api.py) file.
