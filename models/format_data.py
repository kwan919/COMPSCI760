import pandas as pd
from pandas import DataFrame
import numpy as np

class Restructure():
    def __init__(self, csv_data: DataFrame) -> None:
        self.csv_data = csv_data
    
    def process(self, index_name: str, feature_name: str, new_feature_name: str) -> DataFrame:
        """ transpose the raw data to feature data"""
        new_data = np.array([])
        index_list = np.array([])
        for index, group in self.csv_data.groupby(index_name):
            index_list = np.append(index_list, index)
            new_data = np.append(new_data, group[feature_name].to_numpy())
        new_data = new_data.reshape((len(index_list), -1))
        new_df = pd.DataFrame(new_data, index=index_list)
        new_df.index.name = index_name
        new_df.columns = [new_feature_name+str(i) for i in range(len(new_data[0]))]
        return new_df
        