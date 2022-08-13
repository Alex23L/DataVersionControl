import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data", sep='\s+',
                 names=['status_of_account', 'duartion_in_months', 'credit_history', 'purpose', 'credit_amount',
                        'savings', 'employed_since', 'Installment_rate_in_percentage', 'Personal_status_and_sex',
                        'other_debtors_or_guarantors', 'present_residence_since', 'property', 'age_in_years',
                        'other_installment_plans', 'housing', 'number_of_credits', 'job', 'number_of_ppl_Maintenance',
                        'telephone', 'foreign_worker', 'class'])

from sklearn.model_selection import train_test_split

dfnummeric = pd.get_dummies(df)

x = dfnummeric.drop(["class"], axis=1)
y = dfnummeric["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

pathbasedata = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\basedata\\'
pathtestdata = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\testdata\\'
pathtraindata = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\traindata\\'

df.to_csv(pathbasedata + 'basedata.csv')
x_train.to_csv(pathtraindata + 'x_train.csv')
y_train.to_csv(pathtraindata + 'y_train.csv')
x_test.to_csv(pathtestdata + 'x_test.csv')
y_test.to_csv(pathtestdata + 'y_test.csv')