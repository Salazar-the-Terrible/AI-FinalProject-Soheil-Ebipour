import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

from preprocess import X, y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(128,),
    activation="relu",
    solver="adam",
    max_iter=25,
    verbose=True
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Test accuracy:", accuracy)

joblib.dump(model, "mlp_model.pkl")
