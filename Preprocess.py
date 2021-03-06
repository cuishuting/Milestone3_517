import numpy as np
from sklearn import preprocessing
import pandas as pd

from sklearn.model_selection import train_test_split

df=pd.read_csv('data/train.csv')
cols=df.columns
X_test = np.loadtxt('data/test.csv', delimiter=',', skiprows=1)[:,1:]
ultimate_train = np.loadtxt('data/train.csv', delimiter=',', skiprows=1)
X_ultimate_train,y_ultimate_train=ultimate_train[:,:-1],ultimate_train[:,-1]
y = df.pop('target')
X = df

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.25)

std_scale = preprocessing.MinMaxScaler().fit(X_train)
x_train_normalized = std_scale.transform(X_train)
x_val_normalized=std_scale.transform(X_val)

std_scale_ultimate=preprocessing.MinMaxScaler().fit(X_ultimate_train)
x_train_ultimate_normalized = std_scale_ultimate.transform(X_ultimate_train)
x_test_normalized=std_scale_ultimate.transform(X_test)


# dump data
df_train=pd.DataFrame(data=x_train_normalized,columns=cols[:-1])
df_val=pd.DataFrame(data=x_val_normalized,columns=cols[:-1])
df_y_train=pd.DataFrame(data=y_train,columns=[cols[-1]])
df_y_val=pd.DataFrame(data=y_val,columns=[cols[-1]])

df_train_ultimate=pd.DataFrame(data=x_train_ultimate_normalized,columns=cols[:-1])
df_y_train_ultimate=pd.DataFrame(data=y_ultimate_train,columns=[cols[-1]])

df_test=pd.DataFrame(data=x_test_normalized,columns=cols[:-1])

df_train.to_pickle("data/normalized_x_train_df.pkl")
df_val.to_pickle("data/normalized_x_val_df.pkl")

df_y_train.to_pickle("data/normalized_y_train_df.pkl")
df_y_val.to_pickle("data/normalized_y_val_df.pkl")

df_train_ultimate.to_pickle("data/normalized_x_ultimate_train_df.pkl")
df_y_train_ultimate.to_pickle("data/normalized_y_ultimate_train_df.pkl")
df_test.to_pickle("data/normalized_x_test_df.pkl")





# X.iloc[X_train] # return dataframe train
# train_data = np.loadtxt('data/train.csv', delimiter=',', skiprows=1)
# x_train = train_data[:, :-1]
# y_train = train_data[:,-1]
# x_test = np.loadtxt('data/test.csv', delimiter=',', skiprows=1)
#
# scaler=preprocessing.MinMaxScaler()
# std_scale = scaler.fit(train_data[:,:-1])
# X_train_normalized = std_scale.transform(x_train)
# y_train=
# y_train_normalized
# X_val = std_scale.transform(X_val)
#
# df=pd.read_csv('data/train.csv')
#
# cols=df.columns
# df=pd.DataFrame(data=normalized_train,columns=cols)
# df.to_pickle("data/normalized_train_df.pkl")