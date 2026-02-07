from sklearn.neural_network import MLPClassifier

def run_mlp(x_train, y_train, x_test):
    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300)
    clf.fit(x_train, y_train)
    return clf.predict(x_test)
