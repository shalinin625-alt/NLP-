# =====================================================
# IntelliReq - Random Forest Model
# =====================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# =====================================================
# 1. LOAD DATASET
# =====================================================

dataset_path = "dataset/IntelliReq_1000_Requirement_Dataset.csv"

print("Loading dataset...")

data = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")

print("Number of rows:", len(data))


# =====================================================
# 2. REMOVE EMPTY VALUES
# =====================================================

data = data.dropna(
    subset=[
        "Requirement",
        "Requirement_Type"
    ]
)


# =====================================================
# 3. INPUT AND OUTPUT
# =====================================================

X = data["Requirement"].astype(str)

y = data["Requirement_Type"].astype(str)


print("\nRequirement classes:")

print(y.value_counts())


# =====================================================
# 4. SPLIT DATASET
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
# 5. TF-IDF
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


print("TF-IDF completed!")


# =====================================================
# 6. TRAIN RANDOM FOREST
# =====================================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)


model.fit(

    X_train_tfidf,

    y_train

)


print("Random Forest training completed!")


# =====================================================
# 7. PREDICTION
# =====================================================

y_pred = model.predict(

    X_test_tfidf

)


# =====================================================
# 8. ACCURACY
# =====================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


print("\n===================================")

print("RANDOM FOREST RESULTS")

print("===================================")

print(

    f"Accuracy: {accuracy * 100:.2f}%"

)


# =====================================================
# 9. CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred

    )

)


# =====================================================
# 10. CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix:")

print(

    confusion_matrix(

        y_test,

        y_pred

    )

)


# =====================================================
# 11. SAVE MODEL
# =====================================================

joblib.dump(

    model,

    "random_forest_model.pkl"

)


joblib.dump(

    vectorizer,

    "random_forest_vectorizer.pkl"

)


print("\n===================================")

print("RANDOM FOREST MODEL SAVED")

print("===================================")

print(

    "Model: random_forest_model.pkl"

)

print(

    "Vectorizer: random_forest_vectorizer.pkl"

)


# =====================================================
# 12. SAMPLE PREDICTION
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