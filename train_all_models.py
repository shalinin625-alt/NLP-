import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# ============================================================
# LOAD DATASET
# ============================================================

DATASET = "IntelliReq_1000_Requirement_Dataset(1).csv"

print("\n======================================")
print("       IntelliReq Model Training")
print("======================================")

df = pd.read_csv(DATASET)

print("Dataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", list(df.columns))


# ============================================================
# CLEAN DATA
# ============================================================

df["Requirement"] = df["Requirement"].fillna("").astype(str)
df["Requirement_Type"] = df["Requirement_Type"].fillna("").astype(str)

X = df["Requirement"]
y = df["Requirement_Type"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    lowercase=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF created successfully!")


# ============================================================
# 1. LOGISTIC REGRESSION
# ============================================================

print("\n--------------------------------------")
print("Training Logistic Regression")
print("--------------------------------------")

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(
    X_train_tfidf,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test_tfidf
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

print(
    "Logistic Regression Accuracy:",
    round(logistic_accuracy * 100, 2),
    "%"
)


# Save model
with open(
    "logistic_regression_model.pkl",
    "wb"
) as f:

    pickle.dump(
        logistic_model,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# Save vectorizer
with open(
    "tfidf_vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


print("Logistic Regression model saved.")


# ============================================================
# 2. SVM
# ============================================================

print("\n--------------------------------------")
print("Training SVM")
print("--------------------------------------")

svm_model = SVC(
    kernel="linear",
    probability=True,
    random_state=42
)

svm_model.fit(
    X_train_tfidf,
    y_train
)

svm_predictions = svm_model.predict(
    X_test_tfidf
)

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

print(
    "SVM Accuracy:",
    round(svm_accuracy * 100, 2),
    "%"
)


# Save SVM model
with open(
    "svm_model.pkl",
    "wb"
) as f:

    pickle.dump(
        svm_model,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# Save SVM vectorizer
with open(
    "svm_tfidf_vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


print("SVM model saved.")


# ============================================================
# 3. RANDOM FOREST
# ============================================================

print("\n--------------------------------------")
print("Training Random Forest")
print("--------------------------------------")

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train_tfidf,
    y_train
)

random_forest_predictions = (
    random_forest_model.predict(
        X_test_tfidf
    )
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

print(
    "Random Forest Accuracy:",
    round(
        random_forest_accuracy * 100,
        2
    ),
    "%"
)


# Save Random Forest model
with open(
    "random_forest_model.pkl",
    "wb"
) as f:

    pickle.dump(
        random_forest_model,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# Save vectorizer
with open(
    "random_forest_vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


print("Random Forest model saved.")


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n======================================")
print("       TRAINING COMPLETED")
print("======================================")

print(
    f"Logistic Regression : "
    f"{logistic_accuracy * 100:.2f}%"
)

print(
    f"SVM                 : "
    f"{svm_accuracy * 100:.2f}%"
)

print(
    f"Random Forest       : "
    f"{random_forest_accuracy * 100:.2f}%"
)

print("\nAll .pkl files regenerated successfully!")

print("======================================")