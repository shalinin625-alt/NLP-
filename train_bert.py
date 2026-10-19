# ============================================================
# IntelliReq - BERT Deep Learning Model
# ============================================================

import pandas as pd
import numpy as np
import torch
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from torch.utils.data import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset_path = "dataset/IntelliReq_1000_Requirement_Dataset.csv"

print("Loading dataset...")

data = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")

print("Number of rows:", len(data))

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# 2. REMOVE EMPTY VALUES
# ============================================================

data = data.dropna(
    subset=[
        "Requirement",
        "Requirement_Type"
    ]
)


# ============================================================
# 3. GET TEXT AND LABEL
# ============================================================

texts = data["Requirement"].astype(str).tolist()

labels = data["Requirement_Type"].astype(str).tolist()


# ============================================================
# 4. ENCODE LABELS
# ============================================================

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(labels)

num_labels = len(label_encoder.classes_)


print("\nRequirement classes:")

for i, label in enumerate(label_encoder.classes_):

    print(i, "=", label)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    texts,

    encoded_labels,

    test_size=0.20,

    random_state=42,

    stratify=encoded_labels

)


print("\nTraining requirements:", len(X_train))

print("Testing requirements:", len(X_test))


# ============================================================
# 6. LOAD BERT TOKENIZER
# ============================================================

print("\nLoading BERT tokenizer...")

tokenizer = BertTokenizer.from_pretrained(

    "bert-base-uncased"

)

print("Tokenizer loaded!")


# ============================================================
# 7. DATASET CLASS
# ============================================================

class RequirementDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer
    ):

        self.texts = texts

        self.labels = labels

        self.tokenizer = tokenizer


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, index):

        text = self.texts[index]

        label = self.labels[index]

        encoding = self.tokenizer(

            text,

            truncation=True,

            padding="max_length",

            max_length=128,

            return_tensors="pt"

        )

        item = {

            key: value.squeeze(0)

            for key, value in encoding.items()

        }

        item["labels"] = torch.tensor(

            label,

            dtype=torch.long

        )

        return item


# ============================================================
# 8. CREATE DATASETS
# ============================================================

train_dataset = RequirementDataset(

    X_train,

    y_train,

    tokenizer

)


test_dataset = RequirementDataset(

    X_test,

    y_test,

    tokenizer

)


# ============================================================
# 9. LOAD PRETRAINED BERT
# ============================================================

print("\nLoading pretrained BERT model...")

model = BertForSequenceClassification.from_pretrained(

    "bert-base-uncased",

    num_labels=num_labels

)

print("BERT model loaded!")


# ============================================================
# 10. METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions = eval_prediction.predictions

    labels = eval_prediction.label_ids

    predictions = np.argmax(

        predictions,

        axis=1

    )

    accuracy = accuracy_score(

        labels,

        predictions

    )

    return {

        "accuracy": accuracy

    }


# ============================================================
# 11. TRAINING ARGUMENTS
# ============================================================

print("\nCreating BERT training settings...")


training_args = TrainingArguments(

    output_dir="./bert_results",

    num_train_epochs=3,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    learning_rate=2e-5,

    weight_decay=0.01,

    logging_steps=10,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    report_to="none"

)


# ============================================================
# 12. CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics

)


# ============================================================
# 13. TRAIN BERT
# ============================================================

print("\n===================================")

print("TRAINING BERT MODEL")

print("===================================")


trainer.train()


print("\nBERT training completed!")


# ============================================================
# 14. EVALUATE
# ============================================================

print("\n===================================")

print("BERT EVALUATION")

print("===================================")


evaluation = trainer.evaluate()

print(evaluation)


# ============================================================
# 15. PREDICTIONS
# ============================================================

print("\nMaking predictions...")


prediction_output = trainer.predict(

    test_dataset

)


predictions = np.argmax(

    prediction_output.predictions,

    axis=1

)


# ============================================================
# 16. ACCURACY
# ============================================================

accuracy = accuracy_score(

    y_test,

    predictions

)


print("\n===================================")

print("BERT RESULTS")

print("===================================")


print(

    f"Accuracy: {accuracy * 100:.2f}%"

)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")


print(

    classification_report(

        y_test,

        predictions,

        target_names=label_encoder.classes_

    )

)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")


print(

    confusion_matrix(

        y_test,

        predictions

    )

)


# ============================================================
# 19. SAVE MODEL
# ============================================================

model_path = "bert_requirement_classifier"


model.save_pretrained(

    model_path

)


tokenizer.save_pretrained(

    model_path

)


joblib.dump(

    label_encoder,

    "bert_label_encoder.pkl"

)


print("\n===================================")

print("BERT MODEL SAVED SUCCESSFULLY")

print("===================================")


print(

    "Model folder:",

    model_path

)


print(

    "Label encoder: bert_label_encoder.pkl"

)


# ============================================================
# 20. SAMPLE PREDICTION
# ============================================================

sample_requirement = [

    "The system shall encrypt user passwords."

]


sample_encoding = tokenizer(

    sample_requirement,

    truncation=True,

    padding=True,

    max_length=128,

    return_tensors="pt"

)


model.eval()


with torch.no_grad():

    output = model(

        **sample_encoding

    )


sample_prediction = torch.argmax(

    output.logits,

    dim=1

).item()


predicted_class = label_encoder.inverse_transform(

    [sample_prediction]

)[0]


print("\n===================================")

print("BERT SAMPLE PREDICTION")

print("===================================")


print(

    "Requirement:",

    sample_requirement[0]

)


print(

    "Predicted Type:",

    predicted_class

)


print("===================================")