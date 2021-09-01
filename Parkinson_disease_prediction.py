import numpy as np
import pandas as pd
import os, sys
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score

#Reading the data
df=pd.read_csv('C:\\Users\\DELL\\Downloads\\work\\Projects data flair\\parkinsons.txt', encoding='latin')
print(df.head(5))

# The column status is our target variable here and the rest are features.

features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df['status'].values

#Counting the number of 0 and 1
print(labels[labels==1].shape[0], labels[labels==0].shape[0])

#Initializing the Maxminscaler.The MinMaxScaler transforms features by scaling them to a given range. 
#Scaling is important to avoid some features having more influence while predicting.
#  The fit_transform() method fits to the data and then transforms it. We don’t need toS scale the labels.
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

#using train_test_split function to split the data for training and testing purposes. We took test size to be 20% of original
#data set. We also took a ranodm_state so that the train and test data do not change everytime we run the code.
X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=7)

#XGB classifier(Extreme Gradient Boosting) is a powerful ensemble learning model widely known for its high speed.  
model=XGBClassifier(max_iter=50,random_state=7)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("The Accuracy of the model is "+str(accuracy_score(y_test,y_pred)*100)+" %")
print("The Precision of the model is "+str(precision_score(y_test,y_pred)*100)+" %")
print("The f1 score of the model is "+str(f1_score(y_test,y_pred)))
print(model.score(X_test,y_pred))