from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import pandas as pd

pathtestdata = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\testdata\\'
pathtraindata = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\traindata\\'
pathscaleddata = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\scaled\\'
pathmodel = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\model\\model.joblib'

x_test = pd.read_csv(pathtestdata + 'x_test.csv')
y_test = pd.read_csv(pathtestdata + 'y_test.csv')

x_train = pd.read_csv(pathtraindata + 'x_train.csv')
y_train = pd.read_csv(pathtraindata + 'y_train.csv')

x_test = x_test.drop(["Unnamed: 0"], axis=1)
y_test = y_test.drop(["Unnamed: 0"], axis=1)


x_train = x_train.drop(["Unnamed: 0"], axis=1)
y_train = y_train.drop(["Unnamed: 0"], axis=1)

print(y_train)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(x_train)
test_scaled = scaler.transform(x_test)

dump(train_scaled, pathscaleddata + 'train_scaler.bin', compress=True)
dump(test_scaled, pathscaleddata + 'test_scaler.bin', compress=True)

model = MLPClassifier(max_iter=1000)
model_trained = model.fit(train_scaled, y_train.values.ravel())

dump(model_trained, pathmodel)