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



""" Sample Output


name  MDVP:Fo(Hz)  MDVP:Fhi(Hz)  MDVP:Flo(Hz)  MDVP:Jitter(%)  MDVP:Jitter(Abs)  MDVP:RAP  MDVP:PPQ  ...     HNR  status      RPDE       DFA   spread1   spread2        D2       
PPE
0  phon_R01_S01_1      119.992       157.302        74.997         0.00784           0.00007   0.00370   0.00554  ...  21.033       1  0.414783  0.815285 -4.813031  0.266482  2.301442  0.284654
1  phon_R01_S01_2      122.400       148.650       113.819         0.00968           0.00008   0.00465   0.00696  ...  19.085       1  0.458359  0.819521 -4.075192  0.335590  2.486855  0.368674
2  phon_R01_S01_3      116.682       131.111       111.555         0.01050           0.00009   0.00544   0.00781  ...  20.651       1  0.429895  0.825288 -4.443179  0.311173  2.342259  0.332634
3  phon_R01_S01_4      116.676       137.871       111.366         0.00997           0.00009   0.00502   0.00698  ...  20.644       1  0.434969  0.819235 -4.117501  0.334147  2.405554  0.368975
4  phon_R01_S01_5      116.014       141.781       110.655         0.01284           0.00011   0.00655   0.00908  ...  19.649       1  0.417356  0.823484 -3.747787  0.234513  2.332180  0.410335

[5 rows x 24 columns]
147 48
C:\Users\DELL\AppData\Local\Programs\Python\Python37\lib\site-packages\xgboost\sklearn.py:892: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
  warnings.warn(label_encoder_deprecation_msg, UserWarning)

[11:56:15] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
The Accuracy of the model is 94.87179487179486 %
The Precision of the model is 96.875 %
The f1 score of the model is 0.96875
1.0 """
