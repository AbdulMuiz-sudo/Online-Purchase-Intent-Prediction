# ============================================
# 1. DATA LOADING
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

data = pd.read_csv("online_shoppers_intention.csv")
print("Shape:", data.shape)
data.info()
print(data.head())

# ============================================
# 2. DATA CLEANING
# ============================================

print("\nMissing values:\n", data.isnull().sum())

data = data.drop_duplicates()
data["Weekend"] = data["Weekend"].astype(int)
data["Revenue"] = data["Revenue"].astype(int)

month_mapping = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
data["Month"] = data["Month"].map(month_mapping)

print("\nAfter cleaning:")
data.info()

# ============================================
# 3. EXPLORATORY ANALYSIS
# ============================================

print(data.describe())

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove("Revenue")

for col in numeric_cols[:6]:
    plt.figure(figsize=(5, 3))
    plt.hist(data[col], bins=30)
    plt.title(col)
    plt.tight_layout()
    plt.show()

for col in numeric_cols[:6]:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=data[col])
    plt.title(col)
    plt.tight_layout()
    plt.show()

category_cols = ["VisitorType", "Weekend"]
for col in category_cols:
    plt.figure(figsize=(5, 3))
    sns.countplot(x=col, data=data)
    plt.title(col)
    plt.tight_layout()
    plt.show()

sns.boxplot(x="Revenue", y="PageValues", data=data)
plt.title("PageValues vs Purchase")
plt.show()

sns.boxplot(x="Revenue", y="BounceRates", data=data)
plt.title("BounceRates vs Purchase")
plt.show()

numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# ============================================
# 4. FEATURE ENGINEERING
# ============================================

data["TotalTimeSpent"] = (
    data["Administrative_Duration"]
    + data["Informational_Duration"]
    + data["ProductRelated_Duration"]
)

features = data.drop(columns=["Revenue"])
labels = data["Revenue"]

numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()

num_transform = Pipeline([("scale", StandardScaler())])
cat_transform = Pipeline([("encode", OneHotEncoder(handle_unknown="ignore"))])

preprocess = ColumnTransformer(
    [
        ("num", num_transform, numeric_features),
        ("cat", cat_transform, categorical_features),
    ]
)

# ============================================
# 5. MODEL TRAINING
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.20, random_state=42, stratify=labels
)

logistic_model = Pipeline(
    [
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)

tree_model = Pipeline(
    [
        ("prep", preprocess),
        (
            "model",
            DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        ),
    ]
)

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model,
}

performance = {}

for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    performance[name] = [acc, prec, rec, auc]

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("AUC:", auc)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(name)
    plt.show()

joblib.dump(logistic_model, "purchase_model.pkl")

# ============================================
# 6. MODEL COMPARISON
# ============================================

perf_df = pd.DataFrame(performance, index=["Accuracy", "Precision", "Recall", "AUC"]).T
print(perf_df)

perf_df.plot(kind="bar", figsize=(8, 4))
plt.title("Model Comparison")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================================
# 7. FEATURE IMPORTANCE
# ============================================

tree_clf = tree_model.named_steps["model"]
encoded_feature_names = numeric_features + list(
    tree_model.named_steps["prep"]
    .named_transformers_["cat"]
    .named_steps["encode"]
    .get_feature_names_out(categorical_features)
)

importances = tree_clf.feature_importances_
top_features = (
    pd.Series(importances, index=encoded_feature_names)
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(8, 4))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Important Features")
plt.show()
