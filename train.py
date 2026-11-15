import os
import pickle
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

output_file = "/home/asus/IsItMutual/models/model.bin"

df = pd.read_csv(
    "/home/asus/IsItMutual/data/Speed Dating Data.csv",
    encoding="ISO-8859-1"
)

# A. Observable profile
profile_features = [
    "gender", "age", "age_o",
    "race", "race_o", "samerace",
    "field_cd", "goal", "date", "go_out"
]

# B. What the participant values in a partner
preference_features = [
    "attr1_1", "sinc1_1", "intel1_1",
    "fun1_1", "amb1_1", "shar1_1"
]

# C. Participant's own impressions after the date
post_date_features = [
    "attr", "sinc", "intel",
    "fun", "amb", "shar",
    "like", "prob", "met"
]

# Combine groups, keeping only existing columns
candidate_features = list(dict.fromkeys(
    profile_features + preference_features + post_date_features
))

candidate_features = [
    col for col in candidate_features
    if col in df.columns
]

X = df[candidate_features].copy()
y = df["match"].copy()


# -----------------------------------
# 1. Create pair-level group IDs
# -----------------------------------

# Check that both participant IDs and wave are available
iid_num = pd.to_numeric(df["iid"], errors="coerce")
pid_num = pd.to_numeric(df["pid"], errors="coerce")
wave_num = pd.to_numeric(df["wave"], errors="coerce")

valid = (
    iid_num.notna()
    & pid_num.notna()
    & wave_num.notna()
)

groups = pd.Series(index=df.index, dtype="object")

# Normalize participant IDs so both directions get the same group
low = np.minimum(iid_num[valid], pid_num[valid]).astype(int)
high = np.maximum(iid_num[valid], pid_num[valid]).astype(int)

# Create pair group IDs
groups.loc[valid] = (
    wave_num[valid].astype(int).astype(str)
    + "_"
    + low.astype(str)
    + "_"
    + high.astype(str)
)

# Give incomplete rows separate group IDs
groups.loc[~valid] = (
    "incomplete_" + df.index[~valid].astype(str)
)

# -----------------------------------
# 2. First split: 60% train, 40% temp
# -----------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.40,
    random_state=42
)

train_idx, temp_idx = next(
    splitter.split(X, y, groups=groups)
)

# -----------------------------------
# 3. Second split: 20% validation, 20% test
# -----------------------------------

X_temp = X.iloc[temp_idx]
y_temp = y.iloc[temp_idx]
groups_temp = groups.iloc[temp_idx]

splitter_temp = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_rel_idx, test_rel_idx = next(
    splitter_temp.split(
        X_temp,
        y_temp,
        groups=groups_temp
    )
)

# Map temporary indices to original indices
val_idx = temp_idx[val_rel_idx]
test_idx = temp_idx[test_rel_idx]

# -----------------------------------
# 4. Create final datasets
# -----------------------------------

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# -----------------------------------
# 5. Preprocessing + tuned Random Forest
# -----------------------------------

categorical_features = [
    "gender", "samerace", "race", "race_o",
    "field_cd", "goal", "date", "go_out", "met"
]

numerical_features = [col for col in candidate_features if col not in categorical_features]


tree_numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

tree_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

tree_preprocessor = ColumnTransformer([
    ("numerical", tree_numerical_pipeline, numerical_features),
    ("categorical", tree_categorical_pipeline, categorical_features)
])

# Best parameters found by GroupKFold + RandomizedSearchCV in the notebook
tuned_rf_model = Pipeline([
    ("preprocessor", tree_preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------------
# 6. Retrain on train + validation (80%)
# -----------------------------------

X_train_val = pd.concat(
    [X_train, X_val],
    axis=0
)

y_train_val = pd.concat(
    [y_train, y_val],
    axis=0
)

final_model = clone(tuned_rf_model)

final_model.fit(X_train_val, y_train_val)

# -----------------------------------
# 7. Evaluate once on the untouched test set
# -----------------------------------

y_test_proba = final_model.predict_proba(X_test)[:, 1]

# Threshold selected using validation data (notebook)
final_threshold = 0.51

y_test_pred = (
    y_test_proba >= final_threshold
).astype(int)

# Threshold-independent metrics
test_ap = average_precision_score(y_test, y_test_proba)
test_roc_auc = roc_auc_score(y_test, y_test_proba)

# Threshold-dependent metrics
test_precision = precision_score(y_test, y_test_pred, zero_division=0)
test_recall = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

print("FINAL TEST RESULTS")
print("-" * 35)
print(f"PR-AUC (Average Precision): {test_ap:.4f}")
print(f"ROC-AUC:                    {test_roc_auc:.4f}")
print(f"Precision:                  {test_precision:.4f}")
print(f"Recall:                     {test_recall:.4f}")
print(f"F1 Score:                   {test_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# -----------------------------------
# 8. Save the model
# -----------------------------------

# The pipeline already contains the imputers and encoder, so no separate
# DictVectorizer is needed. The threshold and feature list are saved with
# the model so the prediction code can reuse them.
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "wb") as f_out:
    pickle.dump(
        {
            "model": final_model,
            "threshold": final_threshold,
            "features": candidate_features
        },
        f_out
    )

print(f"\nThe model is saved to {output_file}")
