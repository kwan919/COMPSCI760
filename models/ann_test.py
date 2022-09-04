from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from numpy import ndarray


class Ann_Test():
    """Test class for ANN
    """

    def __init__(self, layer_size: int, random_state: int, X, y) -> None:
        """Init a ANN model with a layer size and random seed
            Set training feature X and training label y

        Args:
            layer_size (int): set the layer size
            random_state (int): set random state
            X (_type_): training feature
            y (_type_): training label
        """
        self.model = MLPClassifier(hidden_layer_sizes=(
            layer_size,), random_state=random_state)
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
