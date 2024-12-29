"""
@File    :   model_lightning.py
@Time    :   12/2024
@Author  :   nikifori
@Version :   -
"""
import lightning as L
from transformers import DistilBertModel
import torch


class GenreClassifier_lightning(L.LightningModule):
    def __init__(
        self,
        transformer_model: str = "distilbert-base-uncased",
        n_classes: int = 5,

    ):
        super().__init__()
        self.n_classes = n_classes
        self.model = DistilBertModel.from_pretrained(transformer_model)
        self.model_output_length = self.model.transformer.layer[-1].ffn.lin2.out_features
        self.out_linear = torch.nn.Linear(self.model_output_length, self.model_output_length, bias=True)
        self.classification = torch.nn.Linear(self.model_output_length, self.n_classes, bias=True)
        



    def training_step(self): ...

    def validation_step(self): ...

    def test_step(self): ...

    def configure_optimizers(self): ...


def main():
    pass


if __name__ == "__main__":
    main()
