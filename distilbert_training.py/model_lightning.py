"""
@File    :   model_lightning.py
@Time    :   12/2024
@Author  :   nikifori
@Version :   -
"""

import lightning as L
from transformers import DistilBertModel
from transformers import AutoModel
import torch
import numpy as np
from sklearn.metrics import classification_report
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class GenreClassifier_lightning(L.LightningModule):
    def __init__(
        self,
        transformer_model: str = "distilbert-base-uncased",
        exp_folder: Path = None,
        use_activation_func_before_class_layer: bool = True,
    ):
        super().__init__()
        self.use_activation_func_before_class_layer = (
            use_activation_func_before_class_layer
        )
        self.exp_folder = exp_folder
        self.classes_order = ["rap", "pop", "rock", "country", "rb"]
        self.n_classes = len(self.classes_order)
        self.model = AutoModel.from_pretrained(transformer_model).train()

        if not transformer_model in [
            "FacebookAI/roberta-base",
            "google-bert/bert-base-uncased",
        ]:
            self.model_output_length = self.model.transformer.layer[
                -1
            ].ffn.lin2.out_features
        else:
            self.model_output_length = self.model.pooler.dense.out_features

        self.out_linear = torch.nn.Linear(
            self.model_output_length, self.model_output_length, bias=True
        )
        self.classification = torch.nn.Linear(
            self.model_output_length, self.n_classes, bias=True
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        linear_output = self.out_linear(pooled_output)
        if self.use_activation_func_before_class_layer:
            linear_output = torch.nn.functional.gelu(linear_output)
        logits = self.classification(linear_output)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }, batch["label"]
        logits = self.forward(x["input_ids"], x["attention_mask"])
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }, batch["label"]
        logits = self.forward(x["input_ids"], x["attention_mask"])
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Compute metrics
        if batch_idx == 0:
            self.val_preds = preds.cpu().numpy()
            self.val_targets = y.cpu().numpy()
        else:
            self.val_preds = np.concatenate((self.val_preds, preds.cpu().numpy()))
            self.val_targets = np.concatenate((self.val_targets, y.cpu().numpy()))

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Log classification metrics
        report = classification_report(
            self.val_targets,
            self.val_preds,
            target_names=self.classes_order,
            labels=list(range(len(self.classes_order))),
            output_dict=True,
            digits=5,
        )
        self.log_dict(
            {
                f"val_{cls}_precision": report[cls]["precision"]
                for cls in self.classes_order
            }
        )
        self.log_dict(
            {f"val_{cls}_recall": report[cls]["recall"] for cls in self.classes_order}
        )
        self.log_dict(
            {f"val_{cls}_f1": report[cls]["f1-score"] for cls in self.classes_order}
        )
        self.log("val_accuracy", report["accuracy"])

    def test_step(self, batch, batch_idx):
        x, y = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }, batch["label"]
        logits = self.forward(x["input_ids"], x["attention_mask"])
        preds = torch.argmax(logits, dim=1)

        # Compute metrics
        if batch_idx == 0:
            self.test_preds = preds.cpu().numpy()
            self.test_targets = y.cpu().numpy()
        else:
            self.test_preds = np.concatenate((self.test_preds, preds.cpu().numpy()))
            self.test_targets = np.concatenate((self.test_targets, y.cpu().numpy()))

        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        # Log classification metrics
        report = classification_report(
            self.test_targets,
            self.test_preds,
            target_names=self.classes_order,
            labels=list(range(len(self.classes_order))),
            output_dict=True,
            digits=5,
        )
        self.log_dict(
            {
                f"test_{cls}_precision": report[cls]["precision"]
                for cls in self.classes_order
            }
        )
        self.log_dict(
            {f"test_{cls}_recall": report[cls]["recall"] for cls in self.classes_order}
        )
        self.log_dict(
            {f"test_{cls}_f1": report[cls]["f1-score"] for cls in self.classes_order}
        )
        self.log("test_accuracy", report["accuracy"])

        cm = confusion_matrix(
            self.test_targets,
            self.test_preds,
            labels=list(range(len(self.classes_order))),
        )
        cm_str = "\n".join(["\t".join(map(str, row)) for row in cm])

        report_txt = classification_report(
            self.test_targets,
            self.test_preds,
            target_names=self.classes_order,
            labels=list(range(len(self.classes_order))),
            digits=5,
        )

        # Write to file
        with open(self.exp_folder / "test_classification_report.txt", "w") as f:
            f.write(report_txt)
            f.write("\n\nConfusion Matrix:\n")
            f.write("Labels Order: " + ", ".join(self.classes_order) + "\n")
            f.write(cm_str)

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.classes_order
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap="Blues", ax=ax)

        # Save the plot as an image file
        confusion_matrix_image_path = self.exp_folder / "confusion_matrix.png"
        plt.savefig(confusion_matrix_image_path)
        plt.close()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.01)


def main():
    pass


if __name__ == "__main__":
    main()
