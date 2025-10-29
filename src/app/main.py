import os
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"
GENDER_CSV = DATA_DIR / "gender_submission.csv"  
PRED_OUT  = DATA_DIR / "predictions.csv"

ID_COL = "PassengerId"
TARGET = "Survived"
NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Pclass", "Sex", "Embarked"]

def banner(msg):
    print("\n" + "="*len(msg))
    print(msg)
    print("="*len(msg))

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[ERROR] Missing file: {path}\n"
                         f"Place Kaggle Titanic CSVs under src/data/")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {path.name} with shape {df.shape}")
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # strip whitespace in object columns; fill Embarked quick default
    for c in df.select_dtypes(include="object"):
        df[c] = df[c].astype(str).str.strip()
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].replace("", np.nan).fillna("S")
    return df

def build_pipeline():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        [
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])
    return pipe

def main():
    banner("Step 1: Load training data")
    train_df = load_csv(TRAIN_CSV)
    train_df = basic_clean(train_df)

    banner("Step 2: Quick EDA (prints only)")
    print("[EDA] Columns:", list(train_df.columns))
    print("[EDA] Missing values (top 10):")
    print(train_df.isna().sum().sort_values(ascending=False).head(10))
    if TARGET in train_df.columns:
        print(f"[EDA] Overall survival rate: {train_df[TARGET].mean():.3f}")
        if "Sex" in train_df.columns:
            print("[EDA] Survival by Sex:")
            print(train_df.groupby("Sex")[TARGET].mean().round(3))
        if "Pclass" in train_df.columns:
            print("[EDA] Survival by Pclass:")
            print(train_df.groupby("Pclass")[TARGET].mean().round(3))

    banner("Step 3: Feature selection & split")
    needed = [TARGET, ID_COL] + NUM_COLS + CAT_COLS
    missing = [c for c in needed if c not in train_df.columns]
    if missing:
        raise SystemExit(f"[ERROR] train.csv missing columns: {missing}")

    X = train_df[NUM_COLS + CAT_COLS]
    y = train_df[TARGET].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[SPLIT] Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

    banner("Step 4: Build & fit Logistic Regression")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    print("[MODEL] Trained LogisticRegression pipeline")

    banner("Step 5: Metrics on training and validation")
    yhat_tr = pipe.predict(X_train)
    yhat_va = pipe.predict(X_valid)
    p_tr = pipe.predict_proba(X_train)[:,1]
    p_va = pipe.predict_proba(X_valid)[:,1]

    tr_acc = accuracy_score(y_train, yhat_tr)
    va_acc = accuracy_score(y_valid, yhat_va)
    tr_auc = roc_auc_score(y_train, p_tr)
    va_auc = roc_auc_score(y_valid, p_va)

    print(f"[METRIC][TRAIN]  Accuracy={tr_acc:.3f}  AUC={tr_auc:.3f}")
    print(f"[METRIC][VALID]  Accuracy={va_acc:.3f}  AUC={va_auc:.3f}")
    print("[VALID] Classification report:")
    print(classification_report(y_valid, yhat_va, digits=3))

    banner("Step 6: Load test.csv and predict")
    test_df = load_csv(TEST_CSV)
    test_df = basic_clean(test_df)

    # ensure required columns exist
    test_missing = [c for c in [ID_COL] + NUM_COLS + CAT_COLS if c not in test_df.columns]
    if test_missing:
        raise SystemExit(f"[ERROR] test.csv missing columns: {test_missing}")

    X_test = test_df[NUM_COLS + CAT_COLS]
    test_pred = pipe.predict(X_test).astype(int)
    test_proba = pipe.predict_proba(X_test)[:, 1]  
    out = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: test_pred})
    out.to_csv(PRED_OUT, index=False)
    print(f"[OUTPUT] Wrote predictions → {PRED_OUT}")

    banner("Step 7: Evaluate on test.csv")

    gender = load_csv(GENDER_CSV)  
    merged = out.merge(gender[[ID_COL, TARGET]], on=ID_COL, how="left", suffixes=("_pred", "_baseline"))
    # handel missing baseline labels
    missing_lab = merged[TARGET + "_baseline"].isna().sum()
    if missing_lab > 0:
        print(f"[WARN] {missing_lab} rows have no baseline label in gender_submission.csv")

    eval_df = merged.dropna(subset=[TARGET + "_baseline"]).copy()
    y_baseline = eval_df[TARGET + "_baseline"].astype(int).to_numpy()
    y_pred = eval_df[TARGET + "_pred"].astype(int).to_numpy()

    eval_df = eval_df.merge(
        pd.DataFrame({ID_COL: test_df[ID_COL], "proba": test_proba}),
        on=ID_COL, how="left"
    )
    y_score = eval_df["proba"].to_numpy()

    acc_vs_baseline = accuracy_score(y_baseline, y_pred)
    try:
        auc_vs_baseline = roc_auc_score(y_baseline, y_score)
    except ValueError:
        auc_vs_baseline = float("nan")

    print(f"[METRIC][TEST] Accuracy={acc_vs_baseline:.3f}  AUC={auc_vs_baseline:.3f}")


if __name__ == "__main__":
    main()

