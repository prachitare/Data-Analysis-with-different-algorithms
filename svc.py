#importing the library
import numpy as np 
import pandas as pd 

#importing the dataset

db = pd.read_csv('train.csv')



x = db.iloc[:,[0,1,4,8,9,10,15,16,17,18,20,25,27,29,30,31,38,46,49,50,51,53,54,58,62,64,70,71,73,74,77,78,79]].values

y = db.iloc[:, 80]
#print(x)

# no missing values])
# categorical data
#applying label encoding and onehot encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
ohe_x = OneHotEncoder(categorical_features = [13])

x[:,3] = le_x.fit_transform(x[:,3])
x[:,4] = le_x.fit_transform(x[:,4])
x[:,5] = le_x.fit_transform(x[:,5])
x[:,6] = le_x.fit_transform(x[:,6])
x[:,7] = le_x.fit_transform(x[:,7])
x[:,12] = le_x.fit_transform(x[:,12])
x[:,13] = le_x.fit_transform(x[:,13])
x[:,11] = le_x.fit_transform(x[:,11])
x[:,14] = le_x.fit_transform(x[:,14])
x[:,15] = le_x.fit_transform(x[:,15])
x[:,21] = le_x.fit_transform(x[:,21])
x[:,23] = le_x.fit_transform(x[:,23])
x[:,25] = le_x.fit_transform(x[:,25])
x[:,28] = le_x.fit_transform(x[:,28])
x[:,29] = le_x.fit_transform(x[:,29])
x[:,31] = le_x.fit_transform(x[:,31])
x[:,32] = le_x.fit_transform(x[:,32])
 

from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.4, random_state = 0)



#fitting simple linear regression
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)
print(y_pred)


