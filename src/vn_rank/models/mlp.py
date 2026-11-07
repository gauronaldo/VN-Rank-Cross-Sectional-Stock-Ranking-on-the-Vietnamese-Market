"""PyTorch MLP ranking model, the neural-network comparison.

A feed-forward network (2 to 3 hidden layers of equal width, ReLU, dropout) regresses the
per-date rank of the forward return on the feature ranks, centred at 0 with missing
ranks filled like Elastic Net. AdamW supplies the weight decay. With a validation frame,
training stops when the validation L2 loss has not improved for ``patience`` epochs and
the best epoch's weights are kept; without one, it trains for a fixed number of epochs
(the tuned best epoch). Weight initialization, dropout, and batch order all derive from
the seed, so a fit is deterministic for a given seed on one machine.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from vn_rank.config import MLPConfig
from vn_rank.models.base import TARGET_COLUMN, RankingModel
from vn_rank.models.linear import RANK_CENTRE

EPOCHS_KEY = "num_epochs"


@dataclasses.dataclass(frozen=True)
class MLPRun:
    """Tuned parameters, seed, and (for a refit) the number of epochs."""

    params: dict[str, float]
    seed: int
    num_epochs: int | None = None


@dataclasses.dataclass(frozen=True)
class MLPSettings:
    """Fixed training settings and the missing-rank fill value."""

    fixed: MLPConfig
    missing_rank_fill: float


def build_network(n_inputs: int, params: dict[str, float]) -> nn.Sequential:
    """Hidden blocks of ``Linear -> ReLU -> Dropout`` followed by one linear output."""
    width, layers = int(params["hidden_units"]), int(params["n_layers"])
    blocks: list[nn.Module] = []
    for i in range(layers):
        blocks += [nn.Linear(n_inputs if i == 0 else width, width), nn.ReLU()]
        blocks.append(nn.Dropout(params["dropout"]))
    return nn.Sequential(*blocks, nn.Linear(width, 1))


class MLPModel(RankingModel):
    """MLP regressor on centred feature ranks.

    Args:
        features: Feature rank columns.
        run: Tuned parameters, seed, and optional fixed number of epochs.
        settings: Fixed training settings.
        lockbox_start: First lockbox date.
    """

    def __init__(
        self, features: Sequence[str], run: MLPRun, settings: MLPSettings, lockbox_start: dt.date
    ) -> None:
        """Store settings; the network is created by ``fit``."""
        super().__init__(features, lockbox_start)
        self.run, self.settings = run, settings
        self.network: nn.Sequential | None = None
        self.best_epoch = 0
        self._generator = torch.Generator()

    def _inputs(self, frame: pd.DataFrame) -> torch.Tensor:
        values = frame[self.features].fillna(self.settings.missing_rank_fill)
        return torch.from_numpy(values.to_numpy(np.float32) - np.float32(RANK_CENTRE))

    def _tensors(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        return self._inputs(frame), torch.from_numpy(frame[TARGET_COLUMN].to_numpy(np.float32))

    def _train_epoch(
        self,
        net: nn.Sequential,
        optimizer: torch.optim.Optimizer,
        data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = data
        net.train()
        order = torch.randperm(len(x), generator=self._generator)
        for start in range(0, len(x), self.settings.fixed.batch_size):
            batch = order[start : start + self.settings.fixed.batch_size]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(net(x[batch]).squeeze(1), y[batch])
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

    @staticmethod
    def _loss(net: nn.Sequential, x: torch.Tensor, y: torch.Tensor) -> float:
        net.eval()
        with torch.no_grad():
            return float(nn.functional.mse_loss(net(x).squeeze(1), y))

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> None:
        torch.manual_seed(self.run.seed)
        self._generator.manual_seed(self.run.seed)
        net = build_network(len(self.features), self.run.params)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.run.params["learning_rate"],
            weight_decay=self.run.params["weight_decay"],
        )  # fmt: skip
        data = self._tensors(train)
        if valid is None:
            epochs = self.run.num_epochs or self.settings.fixed.max_epochs
            for _ in range(epochs):
                self._train_epoch(net, optimizer, data)
            self.network, self.best_epoch = net, epochs
            return
        self.network = self._fit_early_stopping(net, optimizer, data, self._tensors(valid))

    def _fit_early_stopping(
        self,
        net: nn.Sequential,
        optimizer: torch.optim.Optimizer,
        data: tuple[torch.Tensor, torch.Tensor],
        valid: tuple[torch.Tensor, torch.Tensor],
    ) -> nn.Sequential:
        best_loss, best_state, waited = float("inf"), copy.deepcopy(net.state_dict()), 0
        for epoch in range(1, self.settings.fixed.max_epochs + 1):
            self._train_epoch(net, optimizer, data)
            loss = self._loss(net, *valid)
            if loss < best_loss:
                best_loss, best_state, waited = loss, copy.deepcopy(net.state_dict()), 0
                self.best_epoch = epoch
            else:
                waited += 1
            if waited >= self.settings.fixed.patience:
                break
        net.load_state_dict(best_state)
        return net

    def fitted_attributes(self) -> dict[str, float]:
        """The number of epochs a refit on the whole window should use."""
        return {EPOCHS_KEY: float(max(self.best_epoch, 1))}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return predicted target ranks.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self.network is None:
            raise RuntimeError("MLPModel.predict called before fit")
        self.network.eval()
        with torch.no_grad():
            scores = self.network(self._inputs(frame)).squeeze(1).numpy()
        return np.asarray(scores, dtype=float)
