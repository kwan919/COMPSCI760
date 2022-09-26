import numpy as np

from pandas import DataFrame
from tqdm import tqdm

class Restructure():
    def __init__(self, csv_data: DataFrame) -> None:
        self.csv_data = csv_data

    def to_windows(self, m: int) -> DataFrame:
        """Transform the data from raw to the windowed data
            row size = m x number_of_features, 
                The last data could not form a window will be dropped

        Args:
            m (int): number of features in windows

        Returns:
            DataFrame: The windowed data
        """
        total_length = len(self.csv_data)
        number_of_windows = int(total_length / m)
        columns_name = self.csv_data.columns
        columns_length = len(columns_name)

        data_array = self.csv_data.to_numpy()[:number_of_windows*m, :]
        new_data_array = np.array([])
        for row_matrix in tqdm(np.array_split(data_array, number_of_windows)):
            new_data_array = np.append(
                new_data_array, row_matrix.flatten("F"))
        new_data_array = new_data_array.reshape(-1, m*columns_length)

        new_columns_name = []
        for name in columns_name:
            for i in range(m):
                new_columns_name.append(name + "_" + str(i))

        return DataFrame(new_data_array, columns=new_columns_name)
