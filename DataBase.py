from abc import ABC, abstractmethod
import pandas as pd


class DataBase(ABC):

    directory: str
    fileName: str
    dataFrame: pd.DataFrame

    @abstractmethod
    def loadDatabase(self) -> None:
        pass

    @abstractmethod
    def isAuthentic(self, numId: int, nameId: str) -> bool:
        pass
