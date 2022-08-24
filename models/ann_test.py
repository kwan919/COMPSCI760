from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pandas import DataFrame

class Ann_Test():
    def __init__(self, layer_size, random_state, data: DataFrame) -> None:
        self.model = MLPClassifier(hidden_layer_sizes=(layer_size,), random_state=random_state)
        self.data = data

    def test(self, test_size: float):
        X_train, X_test, y_train, y_test = train_test_split(self.data.iloc[:,0:-1], self.data.iloc[:,-1], test_size=test_size)
        self.model.fit(X_train, y_train)
        predict_label = self.model.predict(X_test)
        return predict_label, y_test