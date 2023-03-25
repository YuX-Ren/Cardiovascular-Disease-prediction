import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import scipy
import seaborn as sea
import plotly.express as px
url = './cardiovascular.txt'
data = pd.read_csv(url,sep=';',decimal=',')
# print(data)
# let's separate index from other columns
data.index = data.iloc[:,0]
df = data.iloc[:,1:]
df.famhist=[i=='Present' for i in df.famhist ]
df = df.astype('float')


# fig, ax = plt.subplots(ncols=1, figsize=(20,10))
# plt.tight_layout(pad=10)
# ax.pie(x=df['chd'].value_counts(), labels=['Cardio', 'No cardio'],autopct='%1.1f%%', shadow=True, startangle=90, explode=(0.05,0.0))
# ax.title.set_text('Cardio percentage')
# plt.show()


# fig, ax = plt.subplots(ncols=1, figsize=(20,10))
# plt.tight_layout(pad=10)
# ax.pie(x=df['famhist'].value_counts(), labels=['famhist', 'No famhist'],autopct='%1.1f%%', shadow=True, startangle=90, explode=(0.05,0.0))
# ax.title.set_text('famhist percentage')
# plt.show()


# fig, ax = plt.subplots(ncols=1, figsize=(20,10))
# plt.tight_layout(pad=18)
# sea.boxplot(data=df, x='chd', y='age', ax=ax)
# ax.title.set_text('Age')
# ax.set_xticklabels(['No-cardio', 'Cardio'])
# ax.set_xlabel("")
# plt.show()


# df = df.drop(['chd'],axis=1)
# df = df.drop(['famhist'],axis=1)
# plt.figure(figsize=(16,5))
# sea.boxplot(data=df)
# plt.title("Distribution of the values of all potential standardized predictors")
# plt.grid()
# plt.show()

# plt.figure(figsize=(16,5))
# sea.heatmap(df.corr(),annot=True,cmap='YlGnBu',fmt='.2f',linewidths=2)
# plt.show()

# plt.figure(figsize=(16,5))
# sea.boxplot(data=df)
# plt.title("Distribution of the values of all potential standardized predictors")
# plt.grid()
# plt.show()