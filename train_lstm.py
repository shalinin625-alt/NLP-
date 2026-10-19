# =====================================================
# IntelliReq - LSTM Deep Learning Model
# =====================================================

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.callbacks import EarlyStopping


# =====================================================
# 1. LOAD DATASET
# =====================================================

dataset_path = "dataset/IntelliReq_1000_Requirement_Dataset.csv"

print("Loading dataset...")

data = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")

print("Number of rows:", len(data))

print("\nColumns:")
print(data.columns.tolist())


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
# 4. ENCODE LABELS
# =====================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nEncoded classes:")

for number, label in enumerate(
    label_encoder.classes_
):

    print(number, "=", label)


# =====================================================
# 5. TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded

)


print("\nTraining requirements:", len(X_train))

print("Testing requirements:", len(X_test))


# =====================================================
# 6. TOKENIZATION
# =====================================================

print("\nCreating tokenizer...")

max_words = 10000

tokenizer = Tokenizer(
    num_words=max_words,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)


X_train_sequences = tokenizer.texts_to_sequences(
    X_train
)

X_test_sequences = tokenizer.texts_to_sequences(
    X_test
)


# =====================================================
# 7. PADDING
# =====================================================

max_length = 40

X_train_padded = pad_sequences(

    X_train_sequences,

    maxlen=max_length,

    padding="post",

    truncating="post"

)


X_test_padded = pad_sequences(

    X_test_sequences,

    maxlen=max_length,

    padding="post",

    truncating="post"

)


print("Tokenization completed!")

print(
    "Training shape:",
    X_train_padded.shape
)

print(
    "Testing shape:",
    X_test_padded.shape
)


# =====================================================
# 8. BUILD LSTM MODEL
# =====================================================

print("\nBuilding LSTM model...")

model = Sequential([

    Embedding(
        input_dim=max_words,
        output_dim=128,
        input_length=max_length
    ),

    LSTM(
        128,
        return_sequences=False
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )

])


# =====================================================
# 9. COMPILE MODEL
# =====================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


print("\nModel summary:")

model.summary()


# =====================================================
# 10. EARLY STOPPING
# =====================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)


# =====================================================
# 11. TRAIN MODEL
# =====================================================

print("\nTraining LSTM model...")

history = model.fit(

    X_train_padded,

    y_train,

    validation_split=0.2,

    epochs=15,

    batch_size=32,

    callbacks=[
        early_stopping
    ],

    verbose=1

)


print("\nLSTM training completed!")


# =====================================================
# 12. PREDICTION
# =====================================================

print("\nMaking predictions...")

y_probability = model.predict(

    X_test_padded,

    verbose=0

)


y_pred = np.argmax(

    y_probability,

    axis=1

)


# =====================================================
# 13. ACCURACY
# =====================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


print("\n===================================")

print("LSTM RESULTS")

print("===================================")

print(

    f"Accuracy: {accuracy * 100:.2f}%"

)


# =====================================================
# 14. CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred,

        target_names=label_encoder.classes_

    )

)


# =====================================================
# 15. CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix:")

print(

    confusion_matrix(

        y_test,

        y_pred

    )

)


# =====================================================
# 16. SAVE MODEL
# =====================================================

model.save(

    "lstm_requirement_classifier.keras"

)


joblib.dump(

    tokenizer,

    "lstm_tokenizer.pkl"

)


joblib.dump(

    label_encoder,

    "lstm_label_encoder.pkl"

)


print("\n===================================")

print("LSTM MODEL SAVED SUCCESSFULLY")

print("===================================")

print(
    "Model: lstm_requirement_classifier.keras"
)

print(
    "Tokenizer: lstm_tokenizer.pkl"
)

print(
    "Label Encoder: lstm_label_encoder.pkl"
)


# =====================================================
# 17. SAMPLE PREDICTION
# =====================================================

sample_requirement = [

    "The system shall encrypt user passwords."

]


sample_sequence = tokenizer.texts_to_sequences(

    sample_requirement

)


sample_padded = pad_sequences(

    sample_sequence,

    maxlen=max_length,

    padding="post",

    truncating="post"

)


sample_probability = model.predict(

    sample_padded,

    verbose=0

)


sample_prediction = np.argmax(

    sample_probability,

    axis=1

)


predicted_class = label_encoder.inverse_transform(

    sample_prediction

)


print("\n===================================")

print("LSTM SAMPLE PREDICTION")

print("===================================")

print(

    "Requirement:",

    sample_requirement[0]

)

print(

    "Predicted Type:",

    predicted_class[0]

)

print("===================================")