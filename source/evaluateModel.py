import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import load, dump


pathtestdata = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\testdata\\'
pathtraindata = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\traindata\\'
pathscaleddata = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\scaled\\'
pathmodel = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\model\\model.joblib'
pathevaluation = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\model\\evaluation.txt'

x_test = pd.read_csv(pathtestdata + 'x_test.csv')
y_test = pd.read_csv(pathtestdata + 'y_test.csv')

x_train = pd.read_csv(pathtraindata + 'x_train.csv')
y_train = pd.read_csv(pathtraindata + 'y_train.csv')

x_test = x_test.drop(["Unnamed: 0"], axis=1)
y_test = y_test.drop(["Unnamed: 0"], axis=1)

x_train = x_train.drop(["Unnamed: 0"], axis=1)
y_train = y_train.drop(["Unnamed: 0"], axis=1)

train_scaled = load(pathscaleddata + 'train_scaler.bin')
test_scaled = load(pathscaleddata + 'test_scaler.bin')

model_trained = load(pathmodel)

print(accuracy_score(y_train, model_trained.predict(train_scaled)))
print(accuracy_score(y_test, model_trained.predict(test_scaled)))

train_accuracy = accuracy_score(y_train, model_trained.predict(train_scaled))
test_accuracy = accuracy_score(y_test, model_trained.predict(test_scaled))

with open(pathevaluation, 'w') as f:
    f.write('Train: ' + str(train_accuracy) + ' Test: ' + str(test_accuracy))
