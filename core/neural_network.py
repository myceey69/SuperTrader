"""Feedforward regression ANN in PyTorch: Input -> 64 -> ReLU -> 32 -> ReLU -> 1.

Adam + MSE, mini-batches, validation tracking, best-validation checkpoint,
fixed seeds, CPU only. The target is standardized with TRAIN statistics
internally (and un-standardized in predict) so lr=0.001 trains well on
small-magnitude returns.
Complexity per epoch: O(n * params).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import nn


class TrainingError(Exception):
    pass


class ReturnNet(nn.Module):
    def __init__(self, n_inputs: int, hidden: Sequence[int] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_inputs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1


class ANNRegressor:
    def __init__(self, hidden: Sequence[int] = (64, 32), learning_rate: float = 0.001,
                 epochs: int = 100, batch_size: int = 32, seed: int = 42):
        if not hidden or any(int(h) < 1 for h in hidden):
            raise ValueError("hidden layer sizes must be positive integers")
        if learning_rate <= 0 or epochs < 1 or batch_size < 1:
            raise ValueError("learning_rate > 0, epochs >= 1, batch_size >= 1 required")
        self.hidden, self.lr, self.epochs = tuple(int(h) for h in hidden), learning_rate, int(epochs)
        self.batch_size, self.seed = int(batch_size), seed
        self.model: Optional[ReturnNet] = None
        self.history = TrainingHistory()
        self._y_mean, self._y_std = 0.0, 1.0

    @staticmethod
    def _validate(x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("X must be a non-empty, finite 2-D array")
        if y is not None and (len(y) != len(x) or not np.isfinite(y).all()):
            raise ValueError("y must be finite and match X")

    def fit(self, x_train, y_train, x_val, y_val,
            progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None) -> TrainingHistory:
        x_train, y_train = np.asarray(x_train, np.float32), np.asarray(y_train, np.float64)
        x_val, y_val = np.asarray(x_val, np.float32), np.asarray(y_val, np.float64)
        self._validate(x_train, y_train)
        self._validate(x_val, y_val)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self._y_mean, self._y_std = float(y_train.mean()), float(y_train.std() or 1.0)
        xt = torch.from_numpy(x_train)
        yt = torch.from_numpy(((y_train - self._y_mean) / self._y_std).astype(np.float32))
        xv = torch.from_numpy(x_val)
        yv = torch.from_numpy(((y_val - self._y_mean) / self._y_std).astype(np.float32))
        self.model = ReturnNet(x_train.shape[1], self.hidden)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        self.history = TrainingHistory()
        best_val, best_state = float("inf"), copy.deepcopy(self.model.state_dict())
        for epoch in range(1, self.epochs + 1):
            if should_stop and should_stop():
                break
            self.model.train()
            perm = torch.randperm(len(xt), generator=gen)
            total = 0.0
            for i in range(0, len(xt), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model(xt[idx]), yt[idx])
                if not torch.isfinite(loss):
                    raise TrainingError(f"Non-finite loss at epoch {epoch}; lower the learning rate.")
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            tr = total / len(xt)
            self.model.eval()
            with torch.no_grad():
                va = loss_fn(self.model(xv), yv).item()
            if not np.isfinite(va):
                raise TrainingError(f"Non-finite validation loss at epoch {epoch}.")
            self.history.train_loss.append(tr)
            self.history.val_loss.append(va)
            if va < best_val:
                best_val, best_state, self.history.best_epoch = va, copy.deepcopy(self.model.state_dict()), epoch
            if progress_cb:
                progress_cb(epoch, self.epochs, tr, va)
        self.model.load_state_dict(best_state)   # restore best-validation weights
        return self.history

    def predict(self, x) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained")
        x = np.asarray(x, np.float32)
        self._validate(x)
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(x)).numpy().astype(np.float64)
        return out * self._y_std + self._y_mean
