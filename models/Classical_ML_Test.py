from pandas import DataFrame
from ann_test import Ann_Test
from knn_test import Knn_test
from decision_tree_test import Decision_Tree_test
from random_forest_test import Random_Forest_test


class CML_Test():
    def __init__(self, random_seed: int, X, y) -> None:
        """Init the classical machine learning pipeline

        Args:
            random_seed (int): set the global random seed for all learning methods
            X (_type_): training feature
            y (_type_): training label
        """

        self.ann_test = Ann_Test(10, random_seed, X, y)
        self.knn_test = Knn_test(random_seed, X, y)
        self.decision_tree_test = Decision_Tree_test(random_seed, X, y)
        self.random_forest_test = Random_Forest_test(random_seed, X, y)

    def all_test(self, cv: int) -> DataFrame:
        result_dict = {"ANN": self.ann_test.cv_test(cv),
                       "KNN": self.knn_test.cv_test(cv),
                       "Decision_Tree": self.decision_tree_test.cv_test(cv),
                       "Random_Forest": self.random_forest_test.cv_test(cv)}

        return DataFrame(result_dict)
