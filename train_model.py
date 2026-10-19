# =====================================================
# IntelliReq - Logistic Regression Model
# =====================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =====================================================
# 1. LOAD DATASET
# =====================================================

dataset_path = "dataset/IntelliReq_1000_Requirement_Dataset.csv"

print("Loading dataset...")

data = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")

print("Number of rows:", len(data))

print("\nColumns:")
print(data.columns.tolist())


# =====================================================
# 2. CHECK REQUIRED COLUMNS
# =====================================================

required_columns = [
    "Requirement",
    "Requirement_Type"
]

for column in required_columns:

    if column not in data.columns:

        print(
            f"\nERROR: Column '{column}' "
            "was not found in the dataset."
        )

        exit()


# =====================================================
# 3. REMOVE EMPTY VALUES
# =====================================================

data = data.dropna(
    subset=[
        "Requirement",
        "Requirement_Type"
    ]
)


# =====================================================
# 4. INPUT AND OUTPUT
# =====================================================

X = data["Requirement"].astype(str)

y = data["Requirement_Type"].astype(str)


print("\nRequirement classes:")

print(
    y.value_counts()
)


# =====================================================
# 5. SPLIT DATASET
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\nTraining requirements:", len(X_train))

print("Testing requirements:", len(X_test))


# =====================================================
# 6. TF-IDF
# =====================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    ngram_range=(1, 2),

    max_features=5000

)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print("TF-IDF feature creation completed.")


# =====================================================
# 7. LOGISTIC REGRESSION
# =====================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(

    max_iter=2000,

    random_state=42

)


model.fit(

    X_train_tfidf,

    y_train

)


print("Model training completed!")


# =====================================================
# 8. PREDICTION
# =====================================================

y_pred = model.predict(
    X_test_tfidf
)


# =====================================================
# 9. ACCURACY
# =====================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


print("\n===================================")

print("LOGISTIC REGRESSION RESULTS")

print("===================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# =====================================================
# 10. CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =====================================================
# 11. CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =====================================================
# 12. SAVE MODEL
# =====================================================

model_path = "logistic_regression_model.pkl"

vectorizer_path = "tfidf_vectorizer.pkl"


joblib.dump(

    model,

    model_path

)


joblib.dump(

    vectorizer,

    vectorizer_path

)


print("\n===================================")

print("MODEL SAVED SUCCESSFULLY")

print("===================================")

print(
    f"Model: {model_path}"
)

print(
    f"Vectorizer: {vectorizer_path}"
)


# =====================================================
# 13. TEST WITH SAMPLE REQUIREMENT
# =====================================================

sample_requirement = [

    "The system shall encrypt user passwords."

]


sample_tfidf = vectorizer.transform(

    sample_requirement

)


prediction = model.predict(

    sample_tfidf

)


print("\n===================================")

print("SAMPLE PREDICTION")

print("===================================")

print(
    "Requirement:",
    sample_requirement[0]
)

print(
    "Predicted Type:",
    prediction[0]
)

print("===================================")