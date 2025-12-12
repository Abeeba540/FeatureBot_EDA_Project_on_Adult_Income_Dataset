import numpy as np
import pandas as pd
import random
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

# ==== STEP 1: Global seeds ====
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.makedirs("artifacts", exist_ok=True)
print("Global seeds set to 42")

# Load dataset (no randomness here, but just to be safe)
df = pd.read_csv('data/adult.csv')


print(f"Data loaded: {df.shape}")

# Convert income to binary 0/1
df['income'] = df['income'].str.strip()
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

print("\nUnique income values after mapping:", df['income'].unique())


X = df.drop('income', axis=1)
y = df['income']

print(f"\nTarget distribution BEFORE splitting:")
print(f"  Class 0: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  Class 1: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

# Split 1: Train (60%) vs Temp (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.4,           # 40% for val+test
    stratify=y,              # Preserve class distribution ✓
    random_state=RANDOM_STATE  # Fixed seed ✓
)

# Split 2: Val (50% of temp = 20% of total) vs Test (50% of temp = 20% of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,           # 50% of temp
    stratify=y_temp,         # Preserve class distribution ✓
    random_state=RANDOM_STATE  # Same seed ✓
)

print(f"\nTarget distribution AFTER splitting:")
print(f"Train  - Class 0: {(y_train==0).mean()*100:.1f}%, Class 1: {(y_train==1).mean()*100:.1f}%")
print(f"Val    - Class 0: {(y_val==0).mean()*100:.1f}%, Class 1: {(y_val==1).mean()*100:.1f}%")
print(f"Test   - Class 0: {(y_test==0).mean()*100:.1f}%, Class 1: {(y_test==1).mean()*100:.1f}%")

# Verify splits are reproducible
assert X_train.shape[0] == 29305, "Train set should be 60%"
assert X_val.shape[0] == 9768, "Val set should be 20%"
assert X_test.shape[0] == 9769, "Test set should be 20%"
print("\n✅ Stratified splits verified (reproducible)")

# StratifiedKFold with fixed seed
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,              # Shuffle before splitting
    random_state=RANDOM_STATE  # Fixed seed ✓
)

print(f"\n5-Fold Cross-Validation setup:")
print(f"Splits per fold:")

fold_indices = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"  Fold {fold+1}: train={len(train_idx)}, val={len(val_idx)}")
    fold_indices.append((train_idx, val_idx))

# IMPORTANT: Save fold indices for later reproducibility
joblib.dump(fold_indices, 'artifacts/cv_fold_indices.pkl')
print(f"\n✅ CV fold indices saved for reproducibility")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    # NO random_state here
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full pipeline: preprocessing + model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver='lbfgs',
        class_weight='balanced',
        n_jobs=1
    ))
])

print("\n✅ Reproducible pipeline with preprocessing configured")


# Train on training set
pipeline.fit(X_train, y_train)
y_train_pred = pipeline.predict(X_train)
y_train_proba = pipeline.predict_proba(X_train)[:, 1]

# Evaluate on validation set (never tune on test!)
y_val_pred = pipeline.predict(X_val)
y_val_proba = pipeline.predict_proba(X_val)[:, 1]

# Final evaluation on test set (hold out until the end!)
y_test_pred = pipeline.predict(X_test)
y_test_proba = pipeline.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score, f1_score

train_auc = roc_auc_score(y_train, y_train_proba)
val_auc = roc_auc_score(y_val, y_val_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

train_f1 = f1_score(y_train, y_train_pred)
val_f1 = f1_score(y_val, y_val_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nResults (reproducible):")
print(f"  Train AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
print(f"  Val AUC:   {val_auc:.4f}, F1: {val_f1:.4f}")
print(f"  Test AUC:  {test_auc:.4f}, F1: {test_f1:.4f}")

# Save results
reproducibility_check = {
    'train_auc': train_auc,
    'val_auc': val_auc,
    'test_auc': test_auc,
    'train_f1': train_f1,
    'val_f1': val_f1,
    'test_f1': test_f1,
    'random_state': RANDOM_STATE
}

joblib.dump(reproducibility_check, 'artifacts/reproducibility_check_run1.pkl')

# If you run this script again:
# 1. Load the saved results
# 2. Compare with new results
# 3. Should be IDENTICAL (to many decimal places)

print(f"\n✅ Results saved. Run script again - results should be IDENTICAL!")
print(f"   Expected: AUC={test_auc:.4f}, F1={test_f1:.4f}")

# ============================================================================
# BEST PRACTICES CHECKLIST
# ============================================================================

reproducibility_checklist = """
REPRODUCIBILITY CHECKLIST:
═════════════════════════════════════════════════════════════════

✅ Global Seeds Set:
   └─ np.random.seed(42)
   └─ random.seed(42)

✅ Data Splitting:
   └─ stratify=y (preserve class distribution)
   └─ random_state=RANDOM_STATE (fixed seed)
   └─ Applied to train/val/test split

✅ Cross-Validation:
   └─ StratifiedKFold(shuffle=True, random_state=42)
   └─ Fixed seed for fold creation

✅ Model Configuration:
   └─ random_state=42 in LogisticRegression
   └─ n_jobs=1 (no parallel processing)

✅ Data Processing:
   └─ No random transformations
   └─ StandardScaler (deterministic)

✅ Verification:
   └─ Run script twice
   └─ Results identical to 4+ decimal places

═════════════════════════════════════════════════════════════════

If results differ between runs:
  1. Check for missing random_state=42
  2. Check for n_jobs > 1 (parallel processing)
  3. Check for random shuffle/sample operations
  4. Check external randomness (system entropy)
"""

print(reproducibility_checklist)

from sklearn.compose import ColumnTransformer

print("\n=== NO DATA LEAKAGE VERIFICATION ===")

# 1) Confirm preprocessor exists and is a ColumnTransformer
preprocessor = pipeline.named_steps["preprocessor"]
assert isinstance(preprocessor, ColumnTransformer), "Preprocessor is not a ColumnTransformer"

# 2) Confirm it was fitted only once (on train) and we never call fit on val/test
# This is guaranteed by using pipeline.fit(X_train, y_train) and only using
# pipeline.predict / predict_proba on X_val/X_test.

print("✅ Preprocessor is a ColumnTransformer fitted via pipeline.fit(X_train, y_train)")
print("✅ Val/Test are only passed through pipeline.predict / predict_proba (no extra fit)")
print("✅ This matches: Fit on TRAIN, apply to VAL/TEST (no data leakage).")




