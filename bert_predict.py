import torch
import joblib

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)


# ============================================================
# LOAD BERT MODEL
# ============================================================

MODEL_PATH = "bert_requirement_classifier"

print("Loading BERT model...")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

label_encoder = joblib.load("bert_label_encoder.pkl")

model.eval()

print("BERT model loaded successfully!")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_requirement(requirement):

    encoding = tokenizer(
        requirement,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model(
            **encoding
        )

    prediction = torch.argmax(
        output.logits,
        dim=1
    ).item()

    predicted_type = label_encoder.inverse_transform(
        [prediction]
    )[0]

    return predicted_type


# ============================================================
# TEST BERT
# ============================================================

if __name__ == "__main__":

    test_requirement = (
        "The system shall encrypt user passwords."
    )

    print("\nRequirement:")
    print(test_requirement)

    print("\nPredicting...")

    predicted_type = predict_requirement(
        test_requirement
    )

    print("\n===================================")
    print("BERT PREDICTION")
    print("===================================")

    print(
        "Requirement Type:",
        predicted_type
    )

    print("===================================")