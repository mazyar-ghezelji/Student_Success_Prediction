# Student performance: analysis + "success" prediction (binary) + GradeClass prediction (multiclass)
# File path (as provided): /mnt/data/Student_performance_data _.csv

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import matplotlib.pyplot as plt
import joblib

# ----------------------------
# 1) Load + basic checks
# ----------------------------
DATA_PATH = "/mnt/data/Student_performance_data _.csv"
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values per column:\n", df.isna().sum().sort_values(ascending=False).head(20))

# ----------------------------
# 2) Define "success" + targets
# ----------------------------
# Success definition (edit if you want): A or B  -> GradeClass in {0,1}
# GradeClass mapping:
# 0:A, 1:B, 2:C, 3:D, 4:F
df["SUCCESS_AB"] = df["GradeClass"].isin([0, 1]).astype(int)

# IMPORTANT:
# GradeClass is defined directly from GPA. If you include GPA as a feature, prediction becomes trivial.
# For a meaningful "predict success from habits/support", we exclude GPA by default.
EXCLUDE_GPA_FROM_FEATURES = True

# ----------------------------
# 3) Quick EDA (optional but useful)
# ----------------------------
print("\nGradeClass distribution:\n", df["GradeClass"].value_counts().sort_index())
print("\nSuccess(AB) distribution:\n", df["SUCCESS_AB"].value_counts())

# Simple plots
fig, ax = plt.subplots(figsize=(7, 4))
df["GradeClass"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_title("GradeClass Distribution (0=A ... 4=F)")
ax.set_xlabel("GradeClass")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
df["StudyTimeWeekly"].plot(kind="hist", bins=20, ax=ax)
ax.set_title("StudyTimeWeekly Distribution")
ax.set_xlabel("Hours/week")
plt.tight_layout()
plt.show()

# ----------------------------
# 4) Feature columns
# ----------------------------
target_multiclass = "GradeClass"
target_binary = "SUCCESS_AB"

# Drop identifiers + target columns
drop_cols = ["StudentID", target_multiclass, target_binary]
if EXCLUDE_GPA_FROM_FEATURES and "GPA" in df.columns:
    drop_cols.append("GPA")

X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y_multi = df[target_multiclass].astype(int)
y_bin = df[target_binary].astype(int)

# Treat coded category-like columns as categorical for one-hot encoding
# (even though they are integers)
categorical_cols = [c for c in [
    "Gender", "Ethnicity", "ParentalEducation", "Tutoring",
    "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"
] if c in X.columns]

# Remaining columns are numeric
numeric_cols = [c for c in X.columns if c not in categorical_cols]

print("\nCategorical cols:", categorical_cols)
print("Numeric cols:", numeric_cols)

# ----------------------------
# 5) Preprocessing pipeline
# ----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ],
    remainder="drop"
)

# ----------------------------
# 6) Train/test split
# ----------------------------
# For multiclass
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# For binary
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# ----------------------------
# 7) Models
# ----------------------------
# Binary: Logistic Regression (strong baseline) + RandomForest + HistGB
bin_models = {
    "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=400, random_state=42, class_weight="balanced_subsample"
    ),
    "HistGB": HistGradientBoostingClassifier(random_state=42)
}

# Multiclass: Logistic Regression + RandomForest + HistGB
multi_models = {
    "LogReg": LogisticRegression(max_iter=3000, class_weight="balanced", multi_class="auto"),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight="balanced_subsample"
    ),
    "HistGB": HistGradientBoostingClassifier(random_state=42)
}

# ----------------------------
# 8) Helper: evaluate binary
# ----------------------------
def eval_binary(name, model, X_train, y_train, X_test, y_test):
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    print(f"\n=== Binary SUCCESS_AB | {name} ===")
    print(classification_report(y_test, preds, digits=3))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Success", "Success"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix (Binary) - {name}")
    plt.tight_layout()
    plt.show()

    if proba is not None:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.4f}")
        RocCurveDisplay.from_predictions(y_test, proba)
        plt.title(f"ROC Curve - {name}")
        plt.tight_layout()
        plt.show()

    # Cross-val ROC-AUC
    if proba is not None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        print(f"CV ROC-AUC: mean={auc_scores.mean():.4f}, std={auc_scores.std():.4f}")

    return pipe

# ----------------------------
# 9) Helper: evaluate multiclass
# ----------------------------
def eval_multiclass(name, model, X_train, y_train, X_test, y_test):
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    print(f"\n=== Multiclass GradeClass | {name} ===")
    print(classification_report(y_test, preds, digits=3))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["A(0)", "B(1)", "C(2)", "D(3)", "F(4)"])
    disp.plot(values_format="d", xticks_rotation=0)
    plt.title(f"Confusion Matrix (Multiclass) - {name}")
    plt.tight_layout()
    plt.show()

    # Cross-val macro F1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"CV F1-macro: mean={f1_scores.mean():.4f}, std={f1_scores.std():.4f}")

    return pipe

# ----------------------------
# 10) Run training + pick best
# ----------------------------
trained_bin = {}
for name, model in bin_models.items():
    trained_bin[name] = eval_binary(name, model, X_train_b, y_train_b, X_test_b, y_test_b)

trained_multi = {}
for name, model in multi_models.items():
    trained_multi[name] = eval_multiclass(name, model, X_train_m, y_train_m, X_test_m, y_test_m)

# ----------------------------
# 11) Feature importance (works best for tree models)
# ----------------------------
def show_rf_feature_importance(rf_pipe, top_n=20, title="Feature importance"):
    # Get feature names after preprocessing
    pre = rf_pipe.named_steps["preprocess"]
    model = rf_pipe.named_steps["model"]

    # Numeric names
    num_names = numeric_cols

    # OneHot names
    cat_ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_ohe.get_feature_names_out(categorical_cols)) if categorical_cols else []

    feature_names = num_names + cat_names
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("Model does not have feature_importances_.")
        return

    order = np.argsort(importances)[::-1][:top_n]
    top = pd.DataFrame({
        "feature": np.array(feature_names)[order],
        "importance": importances[order]
    })

    print("\nTop features:\n", top)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.show()

# Try importance for RandomForest models
if "RandomForest" in trained_bin:
    show_rf_feature_importance(
        trained_bin["RandomForest"],
        title="Top Features (Binary Success) - RandomForest"
    )

if "RandomForest" in trained_multi:
    show_rf_feature_importance(
        trained_multi["RandomForest"],
        title="Top Features (Multiclass GradeClass) - RandomForest"
    )

# ----------------------------
# 12) Save a chosen model (edit which one you prefer)
# ----------------------------
BEST_BINARY_MODEL_NAME = "LogReg"       # change to "RandomForest" or "HistGB" if it performs better
BEST_MULTI_MODEL_NAME = "RandomForest"  # change to whichever looks best in your outputs

joblib.dump(trained_bin[BEST_BINARY_MODEL_NAME], "best_success_binary_model.joblib")
joblib.dump(trained_multi[BEST_MULTI_MODEL_NAME], "best_gradeclass_multiclass_model.joblib")
print("\nSaved models:",
      "best_success_binary_model.joblib, best_gradeclass_multiclass_model.joblib")

# ----------------------------
# 13) Predict on new data (example)
# ----------------------------
# new_students = pd.DataFrame([
#     {"Age": 17, "Gender": 1, "Ethnicity": 2, "ParentalEducation": 3,
#      "StudyTimeWeekly": 12, "Absences": 3, "Tutoring": 0, "ParentalSupport": 3,
#      "Extracurricular": 1, "Sports": 0, "Music": 1, "Volunteering": 0}
# ])
# p_success = trained_bin[BEST_BINARY_MODEL_NAME].predict_proba(new_students)[:,1]
# p_grade = trained_multi[BEST_MULTI_MODEL_NAME].predict(new_students)
# print("P(success A/B):", p_success)
# print("Pred GradeClass:", p_grade)