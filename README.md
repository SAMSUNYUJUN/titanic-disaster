# Titanic
Titanic — Reproducible README (Download Data + Run with Docker in Python and R)

# Repo Structure
titanic-disaster/
├─ src/
│  ├─ app/                  # Python entry (main.py)
│  ├─ r_app/                # R entry (main.R, R Dockerfile, install_packages.R)
│  └─ data/                 # Place Kaggle CSVs here (not in git)
├─ .gitignore
├─ requirements.txt         # Python deps
├─ Dockerfile               # Python Dockerfile
└─ README.md

# Steps to follow
1) Clone the Repository
git clone https://github.com/SAMSUNYUJUN/titanic-disaster.git
cd titanic-disaster


2) Download the Data
2.1 create data folder if not exist
-mkdir -p src/data

2.2 Download 
- train.csv
- test.csv
- gender_submission.scv
from https://www.kaggle.com/competitions/titanic/code, and place the three csv files in the titanic-disaster/src/data


3) Run with Docker — Python App
3.1 Build the image
- docker build -t titanic-app .

3.2 Run the container (mount the local data folder)
- macOS/Linux: docker run --rm -v "$PWD/src/data:/app/src/data" titanic-app
- Windows PowerShell: docker run --rm -v "${PWD}\src\data:/app/src/data" titanic-app

What happens:

The app prints step-by-step logs:

EDA (columns, missing values, survival by Sex/Pclass),

data splitting and model training (Logistic Regression),

training/validation Accuracy and AUC,

predictions on the test set,

comparison vs. gender_submission.csv on matching PassengerId (reported Accuracy/AUC).

A file src/data/predictions.csv is created.


4) Run with Docker — R App
4.1 Build the image
- docker build -f src/r_app/Dockerfile -t titanic-r .

4.2 Run the container (mount the local data folder)
- macOS: docker run --rm -v "$PWD/src/data:/app/src/data" titanic-r
- Windows: docker run --rm -v "${PWD}\src\data:/app/src/data" titanic-r

What happens:

The R script mirrors the Python flow:

prints EDA,

trains a glm (binomial logit),

reports training/validation Accuracy and AUC,

predicts on test.csv,

compares predictions vs. gender_submission.csv (baseline) by PassengerId.

A file src/data/predictions_r.csv is created.


5) Expected Outputs

After running both containers, you should see in src/data/:

predictions.csv       # from Python app
predictions_r.csv     # from R app

