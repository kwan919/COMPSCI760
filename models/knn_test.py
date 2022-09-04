from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from numpy import ndarray


class Knn_test():
    """Test class for K nearest neighbors
    """

    def __init__(self, random_state: int, X, y) -> None:
        """Init a basic K nearest neighbors model
            Set training feature X and training label y

        Args:
            random_state (int): set random state
            X (_type_): training feature
            y (_type_): training label
        """
        self.model = KNeighborsClassifier(random_state=random_state)
        self.X = X
        self.y = y

    def cv_test(self, cv: int) -> ndarray:
        """Run the model with cross validation and return scores

        Args:
            cv (int): number of split on the cross validation

        Returns:
            ndarray: scores list
        """
        scores = cross_val_score(self.model, self.X, self.y, cv=cv)
        return scores
