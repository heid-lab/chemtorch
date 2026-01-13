import pandas as pd
try:
    # Python ≥ 3.12
    from typing import override  # type: ignore
except ImportError:
    # Python < 3.12
    from typing_extensions import override  # type: ignore

from chemtorch.components.data_pipeline.data_source.abstract_data_source import AbstractDataSource


class SingleCSVSource(AbstractDataSource):
    def __init__(
        self,
        data_path: str,
    ):
        self.data_path = data_path

    @override
    def load(self) -> pd.DataFrame:
        """
        Load data from a single CSV file.
        """
        return pd.read_csv(self.data_path)
