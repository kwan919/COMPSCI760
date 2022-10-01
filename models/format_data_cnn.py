import numpy as np

from pandas import DataFrame
from tqdm import tqdm

class Restructure():
    def __init__(self, csv_data: DataFrame) -> None:
        self.csv_data = csv_data

    def to_windows(self, m: int, ) -> DataFrame:
        """Transform the data from raw to the windowed data
            row size = m x number_of_features, 
                The last data could not form a window will be dropped

        Args:
            m (int): number of features in windows

        Returns:
            sig: np matrix 
        """
        total_length = len(self.csv_data)
        number_of_windows = int(total_length / m)
        columns_name = self.csv_data.columns
        columns_length = len(columns_name)

        data_array = self.csv_data.to_numpy()[:number_of_windows*m, :]
        new_data_array = []
        for row_matrix in tqdm(np.array_split(data_array, number_of_windows)):
            reshape_array = np.empty((columns_length, int(np.sqrt(m)), int(np.sqrt(m))))
            for col in range(columns_length): 
                reshape_signal = row_matrix[:, col].reshape(int(np.sqrt(m)), int(np.sqrt(m)))
                reshape_array[col, :, :] = reshape_signal
            new_data_array.append(reshape_array)
        new_data_array = np.stack(new_data_array, axis=0)

        return new_data_array