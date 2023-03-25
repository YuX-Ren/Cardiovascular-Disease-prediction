import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import scipy
import seaborn as sea
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# 数据处理 将famhist P/A 转化为 0/1 bool量 并将 pre 分离

url = './cardiovascular.txt'
data = pd.read_csv(url,sep=';',decimal=',')
# print(data)
# let's separate index from other columns
data.index = data.iloc[:,0]
df = data.iloc[:,1:]
df.famhist=[i=='Present' for i in df.famhist ]
df = df.astype('float')
df = df.drop(['chd'],axis=1)
df=StandardScaler().fit_transform(df)
# sbp	tobacco	ldl	adiposity	famhist	typea	obesity	alcohol	age
df=pd.DataFrame(df,columns=['sbp', 'tobacco', 'ldl', 'adiposity','famhist','typea', 'obesity','alcohol','age'])
df.index=data.index

acc = pd.DataFrame(columns=['lin-reg','LDA','QDA','Fisher\'s LD','Logistic regression','nearest neighbors','decision trees','AdaBoost.M1','RandomForestClassifier'],dtype='float')

# PCA
# pca=PCA(n_components=1) # split in 5 components

# principalComponents = pca.fit_transform(df)

# factors_Df = pd.DataFrame(data = principalComponents)#columns =['PC1','PC2','PC3','PC4','PC5'])

# factors_Df.index=df.index
# df=factors_Df
# print(factors_Df)



y=data['chd']



## lin-reg
num_folds = 20
# Create an instance of the KFold class
kf = KFold(n_splits=num_folds)
# Initialize an empty list to store the accuracy scores
acc_scores = []
X=np.array(df)
Y=np.array(y)
acc_scores=np.array([0.0]*20)
i=0
# Iterate over the folds
for train_index, test_index in kf.split(X):
    # Split the data into training and test sets
    X_Train, X_Test = X[train_index,:], X[test_index,:]
    y_Train, y_Test = Y[train_index], Y[test_index]
    lin_reg = LinearRegression()
    lin_reg.fit(X_Train, y_Train)
    y_pred = lin_reg.predict(X_Test)
# Threshold the predictions to obtain binary class labels
    y_pred_binary = (y_pred > 0.5).astype(int)
    acc_scores[i]= accuracy_score(y_Test, y_pred_binary)
    i=i+1
print("%0.2f accuracy with a standard deviation of %0.2f by lin-reg" % (acc_scores.mean(), acc_scores.std()))
acc['lin-reg']=acc_scores
## lin-reg end


## LDA
# Create an instance of the LinearDiscriminantAnalysis class
lda = LinearDiscriminantAnalysis()
scores = cross_val_score(lda, df, y, cv=20)
print("%0.2f accuracy with a standard deviation of %0.2f by LDA" % (scores.mean(), scores.std()))
acc['LDA']=scores
## LDA end


## QDA end
qda = QuadraticDiscriminantAnalysis()
scores = cross_val_score(qda, df, y, cv=20)
print("%0.2f accuracy with a standard deviation of %0.2f by QDA" % (scores.mean(), scores.std()))
acc['QDA']=scores
## QDA end


## Fisher's LD


# Define the number of folds
num_folds = 20

# Create an instance of the KFold class
kf = KFold(n_splits=num_folds)

# Initialize an empty list to store the accuracy scores
acc_scores = []
X=np.array(df)
Y=np.array(y)
acc_scores=np.array([0.0]*20)
i=0
# Iterate over the folds
for train_index, test_index in kf.split(X):
    # Split the data into training and test sets
    X_Train, X_Test = X[train_index,:], X[test_index,:]
    y_Train, y_Test = Y[train_index], Y[test_index]

    # Split the data into classes
    X_class1 = X_Train[y_Train==1,:]
    X_class2 = X_Train[y_Train==0,:]

    # Compute the mean vectors for each class
    mean_vec1 = np.mean(X_class1, axis=0)
    mean_vec2 = np.mean(X_class2, axis=0)

    # Compute the within-class scatter matrix
    sw = (X_class1 - mean_vec1).T @ (X_class1 - mean_vec1) / X_class1.shape[0] + (X_class2 - mean_vec2).T @ (X_class2 - mean_vec2)/X_class2.shape[0]


    w = np.linalg.inv(sw) @ (mean_vec1 - mean_vec2)
    y_pred = (X_Test-0.5*(mean_vec1 + mean_vec2))@w.T 
    y_pred_binary = (y_pred > 0).astype(int)
    acc_scores[i]= accuracy_score(y_Test, y_pred_binary)
    i=i+1
print("%0.2f accuracy with a standard deviation of %0.2f by Fisher's LD" % (acc_scores.mean(), acc_scores.std()))
acc['Fisher\'s LD']=acc_scores

## Fisher's LD end


## Logistic regression
logis = pd.DataFrame(columns=['Logistic regression','l1','l2'],dtype=float)
logreg = LogisticRegression()
logis['Logistic regression'] = cross_val_score(logreg, df, y, cv=20)
print("%0.2f accuracy with a standard deviation of %0.2f by Logistic regression" % (scores.mean(), scores.std()))


a=np.array([0.0]*100)
for i in range(1,101):
    logreg = LogisticRegression(penalty='l1',solver='liblinear',C=i)
    logis['l1'] = cross_val_score(logreg, df, y, cv=20)
    a[i-1] = cross_val_score(logreg, df, y, cv=20).mean()
plt.figure(figsize=(16,5))
plt.plot(a)
plt.show()

print("%0.2f accuracy with a standard deviation of %0.2f by Logistic regression with l1" % (scores.mean(), scores.std()))
logreg = LogisticRegression(penalty='l2',solver='liblinear')
logis['l2'] = cross_val_score(logreg, df, y, cv=20)
print("%0.2f accuracy with a standard deviation of %0.2f by Logistic regression with l2" % (scores.mean(), scores.std()))
acc['Logistic regression']=logis['Logistic regression']
# Logistic regression end
plt.figure(figsize=(16,5))
sea.boxplot(data=logis)
plt.title("Logistic regression with penalty")
plt.grid()
plt.show()

## nearest neighbors
neigh = KNeighborsClassifier(n_neighbors=39)
scores = cross_val_score(neigh, df, y, cv=20)
print("%0.2f accuracy with a standard deviation of %0.2f by nearest neighbors" % (scores.mean(), scores.std()))
acc['nearest neighbors']=scores
a=np.array([0.0]*100)
for i in range(1,101):
    neigh = KNeighborsClassifier(n_neighbors=i)
    a[i-1] = cross_val_score(neigh, df, y, cv=20).mean()
plt.figure(figsize=(16,5))
plt.plot(a)
plt.show()
## nearest neighbors end


## decision trees
dec = pd.DataFrame(columns=['Gini','entropy'],dtype=float)
dct = tree.DecisionTreeClassifier(criterion="gini")
dec['Gini'] = cross_val_score(dct, df, y, cv=20)
print("%0.2f accuracy with a standard deviation of %0.2f by decision trees with gini" % (scores.mean(), scores.std()))
dct = tree.DecisionTreeClassifier(criterion="entropy")
dec['entropy'] = cross_val_score(dct, df, y, cv=20)
acc['decision trees']=dec['entropy'] 
print("%0.2f accuracy with a standard deviation of %0.2f by decision trees with entropy" % (scores.mean(), scores.std()))
## decision trees end
plt.figure(figsize=(16,5))
sea.boxplot(data=dec)
plt.title("decision trees with different criterion")
plt.grid()
plt.show()


## AdaBoost.M1 
Ada = AdaBoostClassifier(n_estimators=5)
scores = cross_val_score(Ada, df, y, cv=20)
print("%0.2f accuracy with a standard deviation of %0.2f by AdaBoost.M1 " % (scores.mean(), scores.std()))
acc['AdaBoost.M1']=scores
a=[0.0]*25
for i in range(1,26,1):
    Ada = AdaBoostClassifier(n_estimators=i)
    a[i-10] = cross_val_score(Ada, df, y, cv=20).mean()
plt.figure(figsize=(16,5))
plt.plot(a)
plt.show()
## AdaBoost.M1 end


## RandomForestClassifier
a=[0.0]*20
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=5)
scores = cross_val_score(forest, df, y, cv=20)
print("%0.2f accuracy with a standard deviation of %0.2f by RandomForestClassifier " % (scores.mean(), scores.std()))
for i in range(1,21,1):
    forest = RandomForestClassifier( criterion = "gini",n_estimators=i)
    a[i-10] = cross_val_score(forest, df, y, cv=20).mean()
plt.figure(figsize=(16,5))
plt.plot(a)
plt.show()
for i in range(1,21,1):
    forest = RandomForestClassifier( criterion = "entropy",n_estimators=i)
    a[i-10] = cross_val_score(forest, df, y, cv=20).mean()
plt.figure(figsize=(16,5))
plt.plot(a)
plt.show()
acc['RandomForestClassifier']=scores
## RandomForestClassifier end



plt.figure(figsize=(16,5))
sea.boxplot(data=acc)
plt.title("Distribution of the values of all potential standardized predictors")
plt.grid()
plt.show()