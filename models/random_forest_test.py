from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from numpy import ndarray


class Random_Forest_test():
    """Test class for random forest
    """

    def __init__(self, random_state: int, X, y) -> None:
        """Init a basic random forest model
            Set training feature X and training label y

        Args:
            random_state (int): set random state
            X (_type_): training feature
            y (_type_): training label
        """
        self.model = RandomForestClassifier(random_state=random_state)
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
