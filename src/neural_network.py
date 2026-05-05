import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

_SCHEDULERS = {
    "step": torch.optim.lr_scheduler.StepLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    None: None,
}

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        dims = [input_dim]+hidden_dims
        self.network=nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dims[i],dims[i+1]),
                    nn.BatchNorm1d(dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for i in range(len(dims)-1)
            ],
            nn.Linear(dims[-1],2)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TitanicNN(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 15,
        val_fraction: float = 0.15,
        random_state: int = 42,
        optimizer: str = 'adam',
        scheduler: str|None = None,
        scheduler_kwargs: dict|None = None,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        

    def fit(self, X,y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        hidden_dims = self.hidden_dims if self.hidden_dims is not None else [128, 64, 32]

        X=np.array(X,dtype=np.float32)
        y=np.array(y, dtype=np.int64)

        self.classes_ = np.unique(y)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled,
            y, 
            test_size=self.val_fraction, 
            random_state=self.random_state, 
            stratify=y
        )

        input_dim = X_train.shape[1]
        self.model_= MLP(
            input_dim = input_dim,
            hidden_dims = hidden_dims,
            dropout=self.dropout,
        )

        optimizer_cls = _OPTIMIZERS.get(self.optimizer)
        optimizer = optimizer_cls(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler_cls = _SCHEDULERS.get(self.scheduler)
        scheduler_kwargs = self.scheduler_kwargs or {}
        scheduler = scheduler_cls(optimizer, **scheduler_kwargs) if scheduler_cls else None

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
        y_val_t   = torch.tensor(y_val,   dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict | None = None

        for _ in range(self.epochs):
            self.model_.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(X_val_t), y_val_t).item()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self
    
    def predict_proba(self, X) -> np.ndarray:
        X = np.array(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            proba = torch.softmax(self.model_(X_t), dim=1).numpy()

        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]