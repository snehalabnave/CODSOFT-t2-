#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as se


# In[2]:


from warnings import filterwarnings
filterwarnings(action='ignore')


# In[6]:


iris=pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\IRIS.csv")


# In[7]:


iris.head()


# In[8]:


iris.shape


# In[9]:


iris.describe()


# In[10]:


iris.head(150)


# In[13]:


n = len(iris[iris['species'] == 'Iris-versicolor'])
print("No of Versicolor in Dataset:",n)


# In[14]:


n1 = len(iris[iris['species'] == 'Iris-virginica'])
print("No of Versicolor in Dataset:",n1)


# In[15]:


n2 = len(iris[iris['species'] == 'Iris-setosa'])
print("No of Versicolor in Dataset:",n2)


# In[16]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Iris-Versicolor', 'Iris-Setosa', 'Iris-Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()


# In[21]:


#Checking for outliars
import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([iris['sepal_length']])
plt.figure(2)
plt.boxplot([iris['sepal_width']])
plt.show()


# In[22]:


iris.hist()
plt.show()


# In[23]:


iris.plot(kind ='density',subplots = True, layout =(3,3),sharex = False)


# In[24]:


iris.plot(kind ='box',subplots = True, layout =(2,5),sharex = False)


# In[26]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
se.violinplot(x='species',y='petal_length',data=iris)
plt.subplot(2,2,2)
se.violinplot(x='species',y='petal_width',data=iris)
plt.subplot(2,2,3)
se.violinplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,4)
se.violinplot(x='species',y='sepal_width',data=iris)


# In[28]:


se.pairplot(iris,hue='species');


# In[29]:


#Heat Maps
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=se.heatmap(iris.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# In[30]:


X = iris['sepal_length'].values.reshape(-1,1)
print(X)


# In[31]:


Y = iris['sepal_width'].values.reshape(-1,1)
print(Y)


# In[32]:


plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X,Y,color='r')
plt.show()


# In[33]:


#Correlation 
corr_mat = iris.corr()
print(corr_mat)


# In[34]:


from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[35]:


train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[36]:


train_X = train[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
train_y = train.species

test_X = test[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
test_y = test.species


# In[37]:


train_X.head()


# In[38]:


test_y.head()


# In[39]:


#Using LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))


# In[40]:


#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,prediction))


# In[ ]:




