from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Lightweight base model interface for extension."""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
