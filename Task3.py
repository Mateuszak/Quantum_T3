import pandas as pd
from sklearn.neural_network import MLPRegressor

test = pd.read_csv('internship_hidden_test.csv')
train = pd.read_csv("internship_train.csv")

X_train = train.drop(['target'], axis=1)
X_mean = X_train.mean()
X_std = X_train.std()
X_train = (X_train-X_mean)/X_std
X = (test-X_mean)/X_std
y = train['target']

reg = MLPRegressor(max_iter=100, early_stopping=True, hidden_layer_sizes=(128, 64)).fit(X_train, y)
predictions = reg.predict(X)
df = pd.DataFrame(predictions, columns=['target'])
df.to_csv('model_predictions.csv')
