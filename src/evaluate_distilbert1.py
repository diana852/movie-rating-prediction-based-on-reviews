import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import Trainer
from datasets import Dataset

def evaluate_distilbert(trainer: Trainer, dataset: Dataset):
    preds = trainer.predict(dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids

    report = classification_report(
        y_true,
        y_pred,
        labels=[0,1,2,3,4],
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
