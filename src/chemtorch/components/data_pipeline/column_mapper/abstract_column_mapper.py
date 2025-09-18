from abc import ABC, abstractmethod
from typing import Union

import pandas as pd

from chemtorch.utils import DataSplit


class AbstractColumnMapper(ABC):
    @abstractmethod
    def __call__(
        self, data: Union[pd.DataFrame, DataSplit]
    ) -> Union[pd.DataFrame, DataSplit]:
        pass
