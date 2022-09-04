from sklearn import tree
from sklearn.model_selection import cross_val_score
from pandas import DataFrame

from numpy import ndarray


class Decision_Tree_test():
    """Test class for decision tree
    """
    def __init__(self, random_state: int, X, y) -> None:
        """Init a basic decision tree model
            Set training feature X and training label y

        Args:
            random_state (int): set a random seed
            X (_type_): training feature
            y (_type_): training label
        """
        self.model = tree.DecisionTreeClassifier(random_state=random_state)
        self.X = X
        self.y = y

    def test(self, cv: int) -> ndarray:
        """Run the model with cross validation and return scores

        Args:
            cv (int): number of split on the cross validation

        Returns:
            ndarray: scores list
        """
        scores = cross_val_score(self.model, self.X, self.y, cv=cv)
        return scores
