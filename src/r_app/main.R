suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(readr)
  library(pROC)
})

banner <- function(msg) {
  cat("\n", paste(rep("=", nchar(msg)), collapse=""), "\n", msg, "\n",
      paste(rep("=", nchar(msg)), collapse=""), "\n", sep = "")
}

DATA_DIR <- file.path("src", "data")
TRAIN_CSV <- file.path(DATA_DIR, "train.csv")
TEST_CSV  <- file.path(DATA_DIR, "test.csv")
GENDER_CSV <- file.path(DATA_DIR, "gender_submission.csv")
PRED_OUT  <- file.path(DATA_DIR, "predictions_r.csv")

ID_COL <- "PassengerId"
TARGET <- "Survived"
NUM_COLS <- c("Age", "SibSp", "Parch", "Fare")
CAT_COLS <- c("Pclass", "Sex", "Embarked")

load_csv <- function(path) {
  if (!file.exists(path)) {
    stop(sprintf("[ERROR] Missing file: %s\nPlace Kaggle Titanic CSVs under src/data/", path))
  }
  df <- readr::read_csv(path, show_col_types = FALSE)
  cat(sprintf("[INFO] Loaded %s with shape (%d, %d)\n", basename(path), nrow(df), ncol(df)))
  return(df)
}

basic_clean <- function(df) {
  chr_cols <- names(df)[sapply(df, is.character)]
  for (c in chr_cols) df[[c]] <- trimws(df[[c]])
  if ("Embarked" %in% names(df)) {
    df$Embarked[is.na(df$Embarked) | df$Embarked == ""] <- "S"
  }
  df
}

impute_simple <- function(df) {
  # median impute numeric; mode impute categoricals used in formula
  for (c in intersect(NUM_COLS, names(df))) {
    if (anyNA(df[[c]])) {
      med <- median(df[[c]], na.rm = TRUE)
      df[[c]][is.na(df[[c]])] <- med
    }
  }
  for (c in intersect(CAT_COLS, names(df))) {
    if (anyNA(df[[c]])) {
      mode_val <- names(sort(table(df[[c]]), decreasing = TRUE))[1]
      df[[c]][is.na(df[[c]])] <- mode_val
    }
  }
  df
}

# Accuracy helper
acc <- function(y_true, y_pred) mean(y_true == y_pred)

main <- function() {
  banner("Step 1: Load training data")
  train_df <- load_csv(TRAIN_CSV) |> basic_clean()

  banner("Step 2: Quick EDA (prints only)")
  cat("[EDA] Columns: ", paste(names(train_df), collapse=", "), "\n", sep="")
  miss <- sapply(train_df, function(x) sum(is.na(x)))
  miss <- sort(miss, decreasing = TRUE)
  topn <- head(miss, 10)
  cat("[EDA] Missing values (top 10):\n")
  print(topn)
  if (TARGET %in% names(train_df)) {
    cat(sprintf("[EDA] Overall survival rate: %.3f\n",
                mean(train_df[[TARGET]], na.rm = TRUE)))
    if ("Sex" %in% names(train_df)) {
      cat("[EDA] Survival by Sex:\n")
      print(train_df |>
              group_by(Sex) |>
              summarize(rate = mean(.data[[TARGET]], na.rm = TRUE)) |>
              arrange(desc(rate)))
    }
    if ("Pclass" %in% names(train_df)) {
      cat("[EDA] Survival by Pclass:\n")
      print(train_df |>
              group_by(Pclass) |>
              summarize(rate = mean(.data[[TARGET]], na.rm = TRUE)) |>
              arrange(desc(rate)))
    }
  }

  banner("Step 3: Feature selection & split")
  needed <- unique(c(TARGET, ID_COL, NUM_COLS, CAT_COLS))
  missing <- setdiff(needed, names(train_df))
  if (length(missing) > 0) {
    stop(sprintf("[ERROR] train.csv missing columns: %s",
                 paste(missing, collapse=", ")))
  }

  # keep only needed columns; ensure types
  train_df <- train_df[, needed]
  train_df[[TARGET]] <- as.integer(train_df[[TARGET]])
  train_df[[ID_COL]] <- as.integer(train_df[[ID_COL]])
  # factors for categoricals
  for (c in CAT_COLS) if (c %in% names(train_df)) train_df[[c]] <- as.factor(train_df[[c]])

  # simple imputations
  train_df <- impute_simple(train_df)

  # split 80/20 stratified by TARGET
  set.seed(42)
  idx1 <- which(train_df[[TARGET]] == 1)
  idx0 <- which(train_df[[TARGET]] == 0)
  s1 <- sample(idx1, size = floor(0.8 * length(idx1)))
  s0 <- sample(idx0, size = floor(0.8 * length(idx0)))
  tr_idx <- sort(c(s1, s0))
  va_idx <- setdiff(seq_len(nrow(train_df)), tr_idx)

  X_cols <- c(NUM_COLS, CAT_COLS)
  # formula for logistic regression
  fml <- as.formula(paste(TARGET, "~", paste(X_cols, collapse = " + ")))

  banner("Step 4: Build & fit Logistic Regression (glm)")
  model <- glm(fml, data = train_df[tr_idx, ], family = binomial(link = "logit"))
  cat("[MODEL] Trained glm (binomial logit)\n")

  banner("Step 5: Metrics on training and validation")
  # train
  p_tr <- predict(model, newdata = train_df[tr_idx, ], type = "response")
  y_tr <- train_df[[TARGET]][tr_idx]
  yhat_tr <- as.integer(p_tr >= 0.5)
  tr_acc <- acc(y_tr, yhat_tr)
  tr_auc <- tryCatch(as.numeric(pROC::auc(y_tr, p_tr)), error = function(e) NA_real_)
  cat(sprintf("[METRIC][TRAIN]  Accuracy=%.3f  AUC=%.3f\n", tr_acc, tr_auc))

  # valid
  p_va <- predict(model, newdata = train_df[va_idx, ], type = "response")
  y_va <- train_df[[TARGET]][va_idx]
  yhat_va <- as.integer(p_va >= 0.5)
  va_acc <- acc(y_va, yhat_va)
  va_auc <- tryCatch(as.numeric(pROC::auc(y_va, p_va)), error = function(e) NA_real_)
  cat(sprintf("[METRIC][VALID]  Accuracy=%.3f  AUC=%.3f\n", va_acc, va_auc))

  banner("Step 6: Load test.csv and predict")
  test_df <- load_csv(TEST_CSV) |> basic_clean()
  test_needed <- unique(c(ID_COL, NUM_COLS, CAT_COLS))
  test_missing <- setdiff(test_needed, names(test_df))
  if (length(test_missing) > 0) {
    stop(sprintf("[ERROR] test.csv missing columns: %s",
                 paste(test_missing, collapse=", ")))
  }

  test_df <- test_df[, test_needed]
  test_df[[ID_COL]] <- as.integer(test_df[[ID_COL]])
  for (c in CAT_COLS) if (c %in% names(test_df)) test_df[[c]] <- as.factor(test_df[[c]])
  test_df <- impute_simple(test_df)

  p_te <- predict(model, newdata = test_df, type = "response")
  yhat_te <- as.integer(p_te >= 0.5)
  out <- data.frame(PassengerId = test_df[[ID_COL]], Survived = yhat_te)
  readr::write_csv(out, PRED_OUT)
  cat(sprintf("[OUTPUT] Wrote predictions -> %s\n", PRED_OUT))

  banner("Step 7: Evaluate vs gender_submission (baseline)")
  if (file.exists(GENDER_CSV)) {
    gender <- load_csv(GENDER_CSV) %>% select(PassengerId, Survived)
    merged <- out %>% left_join(gender, by = "PassengerId",
                                suffix = c("_pred", "_baseline"))
    missing_lab <- sum(is.na(merged$Survived_baseline))
    if (missing_lab > 0) cat(sprintf("[WARN] %d rows have no baseline label\n", missing_lab))

    eval_df <- merged %>% filter(!is.na(Survived_baseline))
    y_base <- as.integer(eval_df$Survived_baseline)
    y_pred <- as.integer(eval_df$Survived_pred)

    # Align AUC scores with PassengerId order
    eval_df <- eval_df %>% left_join(
      data.frame(PassengerId = test_df[[ID_COL]], proba = p_te),
      by = "PassengerId"
    )
    y_score <- eval_df$proba

    acc_vs_baseline <- mean(y_base == y_pred)
    auc_vs_baseline <- tryCatch(as.numeric(pROC::auc(y_base, y_score)), error = function(e) NA_real_)
    cat(sprintf("[METRIC][TEST vs baseline] Accuracy=%.3f  AUC=%.3f\n",
                acc_vs_baseline, auc_vs_baseline))
  } else {
    cat("[INFO] gender_submission.csv not found; skipping baseline comparison.\n")
  }
}

main()
