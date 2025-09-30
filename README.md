# Deep-Learning-Customer-Churn
Implement Customer Churn using Deep Learning 

# Create Conda Environment
```bash
conda create -n deeplearning-customer-churn python=3.10 -y
```
---
## Activate Conda Environment
```bash
conda activate deeplearning-customer-churn
```
---
## Install Dependicies
```bash
pip install -r requirements.txt
```
---

Initially dataset was corrupted and it hours of time to convert it into other formats and use it
---
Run the model_trainer.py to generate metrics.json
```bash
python src/model_trainer.py
```

Run dvc repro
```bash
dvc repro
```
```bash
dvc metrics add metrics.json
```


DVC ....Add a remote
```bash
dvc remote add -d myremote G:\dvc_remote_storage
```

```bash
dvc push
```
---
## Initialize DVC in your project root:
```bash
dvc init
```

## Add your raw data and processed data to DVC:
```bash
dvc add data/Customer-Churn-Dataset1.csv
dvc add data/processed/train.csv data/processed/test.csv
dvc add saved_models/customer_churn_model.h5
dvc commit

```
## Commit changes to git:
```bash
git add .gitignore *.dvc
git commit -m "Add raw and processed data tracked by DVC"
```

## Configure remote storage (for large files, models):

You can use DagsHub’s remote (or AWS S3, GCP, Azure, etc.)

For DagsHub remote:
```bash
dvc remote add -d origin_dvc dvc://dagshub/<your_username>/<your_repo_name>
dvc push
```

## Run pipeline with:
```bash
dvc repro
```
---
## Create a new repo on DagsHub.

Set DVC remote to DagsHub:
```bash
dvc remote add -d origin_dvc https://dagshub.com/chandrasekharcse522/Deep-Learning-Customer-Churn.dvc

dvc remote modify origin --local auth basic 
dvc remote modify origin --local user chandrasekharcse522 
dvc remote modify origin --local password your-password

dvc remote list
dvc push

```
---
## Push Git repo:
```bash
git remote add origin https://dagshub.com/chandrasekharcse522/Deep-Learning-Customer-Churn.git
git push -u origin main

```
---
## Experiment Tracking:
```bash
dvc repro
git add dvc.yaml dvc.lock
git commit -m "Track metrics.json as metrics for DagsHub"
git push

git add .dvc/config README.md data/.gitignore requirements.txt schema.yaml params.yaml
git add .
git status
git add README.md
git commit -m "Your commit message describing all these changes"
git push

dvc exp run
dvc exp show
dvc exp push origin
git remote -v
dvc exp push origin
```
generate github token and generate dagshub token
```bash
git credential-cache exit
git config --global credential.helper store
dvc exp push origin




```

dvc exp branch naval-weep exp-naval-weep
git checkout exp-naval-weep

git add README.md
git commit -m "Update README.md in exp-naval-weep experiment"

git checkout main
git merge exp-naval-weep

git add README.md
git commit -m "Update README before merging experiment"
git merge exp-naval-weep
git push origin main
git branch -d exp-naval-weep


```bash
uvicorn server_api.app:app --reload
```