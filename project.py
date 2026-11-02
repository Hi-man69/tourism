
# TOURISM MACHINE LEARNING PROJECT

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix
)

# 1. LOAD DATASET

file_path = r"C:\Users\himan\Downloads\India-Tourism-Statistics-2022-Table-2.1.4.csv"

df = pd.read_csv(file_path)

print("\nDATASET INFORMATION")
print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# 2. CLEAN DATASET

df = df.drop_duplicates().copy()

# Clean column names
df.columns = df.columns.str.strip()

# Identify important columns
print("\nColumn names:")
print(df.columns.tolist())

# Find region and nationality columns
region_col = next(
    (c for c in df.columns if c.lower() == "region"), None
)

nationality_col = next(
    (c for c in df.columns if "nationality" in c.lower()), None
)

# Remove aggregate rows if present
for col in [region_col, nationality_col]:
    if col is not None:
        df = df[
            ~df[col].astype(str).str.strip().str.lower().isin(
                ["total", "grand total"]
            )
        ].copy()

# Find arrival columns by year
arrival_cols = {}

for year in [2017, 2018, 2019, 2020, 2021]:
    matches = [
        c for c in df.columns
        if "arrival" in c.lower() and str(year) in c
    ]

    if matches:
        arrival_cols[year] = matches[0]

if 2021 not in arrival_cols:
    raise ValueError(
        "Could not find a 2021 arrivals column. "
        "Check the printed column names."
    )

# Convert arrival data to numbers
for col in arrival_cols.values():
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

print("\nArrival columns:", arrival_cols)

# 3. CORRELATION HEATMAP

numeric_cols = list(arrival_cols.values())

plt.figure(figsize=(9, 6))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap of Tourist Arrivals")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

# ==================================================
# PART A: REGRESSION
# Predict 2021 arrivals using historical arrivals
# ==================================================

# Use only years before 2021 to prevent data leakage
feature_cols = [
    arrival_cols[year]
    for year in [2017, 2018, 2019, 2020]
    if year in arrival_cols
]

target_col = arrival_cols[2021]

reg_df = df[feature_cols + [target_col]].copy()
reg_df = reg_df.dropna(subset=[target_col])

X = reg_df[feature_cols]
y = reg_df[target_col]

if len(reg_df) < 5:
    raise ValueError("Not enough valid rows for regression.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# 4. REGRESSION MODELS

reg_models = {
    "Linear Regression (OLS)": make_pipeline(
        SimpleImputer(strategy="median"),
        LinearRegression()
    ),

    "Polynomial Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression()
    ),

    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeRegressor(
            max_depth=4,
            random_state=42
        )
    ),

    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    ),

    "KNN": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=3)
    ),

    "SVM (SVR)": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVR()
    )
}

reg_results = []
reg_predictions = {}

for name, model in reg_models.items():

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    reg_predictions[name] = predictions

    mse = mean_squared_error(y_test, predictions)

    reg_results.append({
        "Algorithm": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2 Score": r2_score(y_test, predictions)
    })

reg_table = pd.DataFrame(reg_results)
reg_table = reg_table.sort_values("RMSE")

print("\nREGRESSION COMPARISON")
print(reg_table.round(3).to_string(index=False))

reg_table.to_csv("regression_results.csv", index=False)

# 5. REGRESSION COMPARISON GRAPH

plt.figure(figsize=(10, 6))
sns.barplot(
    data=reg_table,
    x="Algorithm",
    y="RMSE"
)
plt.xticks(rotation=30, ha="right")
plt.title("Regression Models Comparison Using RMSE")
plt.ylabel("RMSE (lower is better)")
plt.tight_layout()
plt.savefig("regression_comparison.png", dpi=300)
plt.show()

# 6. ACTUAL VS PREDICTED GRAPH

best_reg_name = reg_table.iloc[0]["Algorithm"]
best_reg_predictions = reg_predictions[best_reg_name]

plt.figure(figsize=(7, 5))
plt.scatter(y_test, best_reg_predictions)

low = min(y_test.min(), best_reg_predictions.min())
high = max(y_test.max(), best_reg_predictions.max())

plt.plot([low, high], [low, high], "--")
plt.xlabel("Actual Tourist Arrivals")
plt.ylabel("Predicted Tourist Arrivals")
plt.title("Actual vs Predicted: " + best_reg_name)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# ==================================================
# PART B: CLASSIFICATION
# Predict region using historical arrival figures
# ==================================================

if region_col is not None:

    class_df = df[[region_col] + feature_cols].copy()
    class_df = class_df.dropna(subset=[region_col])

    # Keep regions with at least two observations
    counts = class_df[region_col].value_counts()
    valid_regions = counts[counts >= 2].index

    class_df = class_df[
        class_df[region_col].isin(valid_regions)
    ].copy()

    if (
        len(class_df) >= 10
        and class_df[region_col].nunique() >= 2
    ):

        class_X = class_df[feature_cols].copy()

        encoder = LabelEncoder()
        class_y = encoder.fit_transform(
            class_df[region_col].astype(str)
        )

        class_counts = pd.Series(class_y).value_counts()

        # Stratified split requires at least two rows per class
        if class_counts.min() >= 2:

            X_train_c, X_test_c, y_train_c, y_test_c = (
                train_test_split(
                    class_X,
                    class_y,
                    test_size=0.25,
                    random_state=42,
                    stratify=class_y
                )
            )

            # 7. CLASSIFICATION MODELS

            class_models = {
                "Logistic Regression": make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(max_iter=2000)
                ),

                "Decision Tree": make_pipeline(
                    SimpleImputer(strategy="median"),
                    DecisionTreeClassifier(
                        max_depth=4,
                        random_state=42
                    )
                ),

                "Random Forest": make_pipeline(
                    SimpleImputer(strategy="median"),
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
                ),

                "KNN": make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    KNeighborsClassifier(n_neighbors=3)
                ),

                "SVM (SVC)": make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    SVC(probability=True, random_state=42)
                )
            }

            class_results = []
            class_predictions = {}

            for name, model in class_models.items():

                model.fit(X_train_c, y_train_c)

                predictions = model.predict(X_test_c)
                probabilities = model.predict_proba(X_test_c)

                class_predictions[name] = predictions

                # Log loss
                loss = log_loss(
                    y_test_c,
                    probabilities,
                    labels=model.classes_
                )

                # ROC-AUC
                try:
                    if len(np.unique(y_test_c)) == 2:
                        auc = roc_auc_score(
                            y_test_c,
                            probabilities[:, 1]
                        )
                    else:
                        auc = roc_auc_score(
                            y_test_c,
                            probabilities,
                            labels=model.classes_,
                            multi_class="ovr",
                            average="macro"
                        )
                except ValueError:
                    auc = np.nan

                class_results.append({
                    "Algorithm": name,
                    "Accuracy": accuracy_score(
                        y_test_c, predictions
                    ),
                    "Precision": precision_score(
                        y_test_c,
                        predictions,
                        average="macro",
                        zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test_c,
                        predictions,
                        average="macro",
                        zero_division=0
                    ),
                    "F1 Score": f1_score(
                        y_test_c,
                        predictions,
                        average="macro",
                        zero_division=0
                    ),
                    "Log Loss": loss,
                    "ROC-AUC": auc
                })

            class_table = pd.DataFrame(class_results)
            class_table = class_table.sort_values(
                "Accuracy", ascending=False
            )

            print("\nCLASSIFICATION COMPARISON")
            print(class_table.round(3).to_string(index=False))

            class_table.to_csv(
                "classification_results.csv",
                index=False
            )

            # 8. CLASSIFICATION ACCURACY GRAPH

            plt.figure(figsize=(9, 5))
            sns.barplot(
                data=class_table,
                x="Algorithm",
                y="Accuracy"
            )
            plt.ylim(0, 1)
            plt.xticks(rotation=25, ha="right")
            plt.title("Classification Accuracy Comparison")
            plt.tight_layout()
            plt.savefig("classification_accuracy.png", dpi=300)
            plt.show()

            # 9. CONFUSION MATRIX

            best_class_name = class_table.iloc[0]["Algorithm"]
            best_class_predictions = class_predictions[
                best_class_name
            ]

            cm = confusion_matrix(
                y_test_c,
                best_class_predictions,
                labels=np.arange(len(encoder.classes_))
            )

            plt.figure(figsize=(9, 7))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Purples",
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_
            )
            plt.title("Confusion Matrix: " + best_class_name)
            plt.xlabel("Predicted Region")
            plt.ylabel("Actual Region")
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig("confusion_matrix.png", dpi=300)
            plt.show()

            print("\nBest classification model:", best_class_name)

        else:
            print(
                "\nClassification skipped: some regions have "
                "too few observations for stratified splitting."
            )

    else:
        print(
            "\nClassification skipped: insufficient data "
            "or fewer than two valid regions."
        )

else:
    print(
        "\nClassification skipped: no Region column was found."
    )

# 10. FINAL SUMMARY

print("\nBEST REGRESSION MODEL:", best_reg_name)
print("\nRegression results saved to regression_results.csv")
print("Project completed successfully!")
