import pandas
import pandas as pd

from pathlib import Path
import os

# df = pd.read_csv(r'C:\Users\Alex\Downloads\2021-parkverstosse.csv1', encoding='latin-1', error_bad_lines=False)

# df = pd.read_csv(r'C:\Users\Alex\Downloads\model.csv', sep='\t', thousands=',' , encoding='utf-8-sig')

# colnames = ['Tattag','Zeit','Tatort','Tatort 2','Tatbestand','Geldbuße']
# df = pd.read_csv(r'C:\Users\Alex\Downloads\pfb11-400it-managementopendatadatenfb32bussgelder-fv2020_bussgelder-fliessender-verkehr-geschwin.csv',sep='\t', thousands=';', names=colnames, encoding='latin-1', header=none)

# df = pd.read_csv(r"C:\Users\Alex\Downloads\model.csv", encoding='latin-1', sep='\t')
# df.write(u'\ufeff'.encode('utf8'))

# df1 = df.copy()

# df['Tattag'] = pd.to_datetime(df['Tattag'], format="%d/%m/%Y", errors='coerce').dt.month

# alex_lsit = df['Tatort'].tolist()

# print(df)


import requests

# data = "https://archive.ics.uci.edu/ml/machine-learning-databases/university/university.data"
# data = requests.get(data)
# temp = data.text
# import re
# fdic = {'def-instance':[], 'state':[]}
# for col in fdic.keys():
#    fdic[col].extend(re.findall(f'\({col} ([^\\\n)]*)' , temp))
# import pandas as pd
# df = pd.DataFrame(fdic)

# print(pd.DataFrame(fdic))


# print(temp)


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



pathdata = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\'
pathdataraw = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\basedata\\'
pathdataprocess = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\data\\process\\'
pathmodel = 'C:\\Users\\Alex\\PycharmProjects\\DVC\\model\\model.joblib'

df.to_csv(pathdataraw + 'basedata.csv')
x_train.to_csv(pathdata + 'x_train.csv')


# import pathlib
# print(pathlib.Path(__file__).parent.resolve())
# print(os.path.basename("data"))
# print(project_root)
# print(output_path)



print(x_train.shape)
print(x_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(x_train)
test_scaled = scaler.transform(x_test)

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(max_iter=1000)
model_trained = model.fit(train_scaled, y_train)

from joblib import dump

dump(model_trained, pathmodel)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_train, model_trained.predict(train_scaled)))
print(accuracy_score(y_test, model_trained.predict(test_scaled)))








# repo_path = Path(__file__).parent.parent
# data_path = repo_path / "data"
# prepared = data_path / "prepared"
#
# def save_as_csv(filenames, destination):
#     data_dictionary = filenames
#     data_frame = pd.DataFrame(filenames,destination)
#     data_frame.to_csv(destination)
#
# save_as_csv(X_train, prepared / "train.csv")

# df_train = X_train.append(y_train, ignore_index= True)