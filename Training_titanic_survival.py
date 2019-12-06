#!/usr/bin/env python
# coding: utf-8

# In[149]:


import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
import sklearn


# In[141]:


#INPUT DATA

data = pd.read_csv('train.csv') 
print(data.head())


# In[17]:


print(data.shape) #Shape of the data frame

data.describe() #statistical data


# In[21]:


#Survived rate 

data["Survived"].value_counts()


# In[28]:


sns.countplot(data["Survived"])


# In[50]:


#Age distribution
sns.countplot(data["Age"])


# In[55]:


# Class Distribution
cols= ["Fare","Sex","Pclass","SibSp","Parch","Embarked"]

#make the subplot size
n_cols=3
n_rows=2
fig, axs=plt.subplots(n_rows,n_cols,figsize=(n_cols*3.2,n_rows*3.2))
for c in range (n_cols):
    for r in range (n_rows):
        i=c*n_rows + r #index to subplot
        ax = axs[r][c]
        sns.countplot(data[cols[i]],hue=data["Survived"],ax=ax)
        ax.legend(loc="upper right")
        
        


# In[73]:


# Survival rate by sex
data.groupby("Sex")[["Survived"]].mean()


# In[81]:


# Survival rate by sex and class
pt=data.pivot_table("Survived",index="Sex",columns="Pclass")
pt.plot()
pt


# In[91]:


# survival rate by Classes

sns.barplot(x="Pclass",y="Survived",data=data)


# In[97]:


# survival rate by sex, age, class
Age=pd.cut(data["Age"],[0,18,80])
data.pivot_table("Survived",["Sex",Age],"Pclass")


# In[103]:


# survival rate by sex, price, class
Fare=pd.cut(data["Fare"],[0,10,20,30,40,50,60,70,80,90,100,150,200,300,400,600])
pt=data.pivot_table("Survived",["Sex",Fare],"Pclass")
pt


# In[107]:


# price paid for each class

plt.scatter(data["Fare"],data["Pclass"],color="blue")

plt.ylabel("Class")
plt.xlabel("Price")
plt.title("Price for each class")
plt.show()


# In[127]:


# check empty values in each  column
data.isna().sum()


# In[118]:


# All values in each column & count 
for val in data:
    print(data[val].value_counts())
    print()


# In[161]:


#Drop columns and rows
data_after=data.drop(["Cabin","PassengerId","Name","Ticket"],axis=1)
Data=data_after.dropna(subset=["Embarked","Age"])

print(Data.shape)
Data.dtypes


# In[162]:


#print unique vals of the columns
print(Data["Sex"].unique())
print(Data["Embarked"].unique())


# In[164]:


from sklearn.preprocessing import LabelEncoder
lablencoder = LabelEncoder()

#Encode Sex column
Data.iloc[:,2] = lablencoder.fit_transform(Data.iloc[:,2].values) #SEX
Data.iloc[:,7] = lablencoder.fit_transform(Data.iloc[:,7].values) #EMBARKED


# In[165]:


#print unique vals of the columns
print(Data["Sex"].unique())
print(Data["Embarked"].unique())


# In[160]:


Data.dtypes


# In[166]:


# split data to X and Y
X=Data.iloc[:,1:8].values
Y=Data.iloc[:,0].values


# In[207]:


# Train and test set 80% : 20%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[208]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[209]:


# Training model

def models(X_train,Y_train):
    
    #Use Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train,Y_train)
    
    #Use Kneighbors
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors = 5,metric="minkowski",p=2)
    knn.fit(X_train,Y_train)
    
    #Use SVC (linear kernel)
    from sklearn.svm import SVC
    svc_lin=SVC(kernel="linear",random_state=0)
    svc_lin.fit(X_train,Y_train)
    
    #Use SVC (RBF kernel)
    from sklearn.svm import SVC
    svc_rbf=SVC(kernel="rbf",random_state=0)
    svc_rbf.fit(X_train,Y_train)
    
    #Use GaussianNB
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train,Y_train)
    
    #Use Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier( criterion="entropy" ,random_state=0)
    tree.fit(X_train,Y_train)
    
    #Use Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
    forest.fit(X_train,Y_train)
    
    #print the accuracy of each model
    print("[0] Logistic Regression Training Accuracy: ",log.score(X_train,Y_train))
    print("[1] K Neighbors Regression Training Accuracy: ",knn.score(X_train,Y_train))
    print("[2] SVC Linear Training Accuracy: ",svc_lin.score(X_train,Y_train))
    print("[3] SVC RBF Training Accuracy: ",svc_rbf.score(X_train,Y_train))
    print("[4] Gaussian NB Training Accuracy: ",gauss.score(X_train,Y_train))
    print("[5] Decision Tree Training Accuracy: ",tree.score(X_train,Y_train))
    print("[6] Random Forest Training Accuracy: ",forest.score(X_train,Y_train))
    
    
    return  log,knn, svc_lin, svc_rbf, gauss, tree, forest
    


# In[210]:


model = models(X_train,Y_train)


# In[211]:


#show the confusion matrix and accuracy for test data
from sklearn.metrics import confusion_matrix
for i in range (len(model)):
    cm= confusion_matrix(Y_test,model[i].predict(X_test))
    

    #Extract  TN, FP ,FN, TP
    TN, FP ,FN, TP =  cm.ravel()
    
    test_score=(TP+TN)/(TN+ FP +FN+ TP)
    print(cm)
    print('Model[',i,'] Testing Accuracy = ',test_score)
    print()
    


# In[226]:


#weights for each feature
i=6
forest=model[i]
importances =pd.DataFrame({"feature":Data.iloc[:,1:8].columns,"importance":np.round(forest.feature_importances_,3)})
importances = importances.sort_values("importance",ascending=False).set_index("feature")

print(importances)
importances.plot.bar()


# In[233]:


#Prediction
pred=model[6].predict(X_test)
print(pred)
print()
print(Y_test)
print()
dif=Y_test-pred
print(dif)


# In[ ]:




