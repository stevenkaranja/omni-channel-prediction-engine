"""PyTorch Lightning training module."""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.transformer_forecaster import DemandForecaster


class ForecastingModule(pl.LightningModule):
    def __init__(self, input_dim: int, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.model = DemandForecaster(
            input_dim=input_dim,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_encoder_layers"],
            pred_horizon=cfg["pred_horizon"],
            dropout=cfg["dropout"],
        )
        self.criterion = nn.HuberLoss(delta=1.0)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        y_hat = self(x)
        return self.criterion(y_hat, y)

    def training_step(self, batch, _):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.cfg["learning_rate"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.cfg["max_epochs"])
        return [opt], [sched]
