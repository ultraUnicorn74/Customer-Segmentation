#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
sns.set()


# In[2]:


data = pd.read_csv('Mall_customers.csv')
data.head()


# In[3]:


data.describe()


# In[4]:


data2 = data.drop(['CustomerID'], axis=1)


# In[5]:


plt.scatter(data2['Age'],data2['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel('Spending Score')


# In[6]:


from sklearn import preprocessing


# In[7]:


x = data2[['Age' ,'Spending Score (1-100)']]


# In[8]:


x_scaled = preprocessing.scale(x)


# In[9]:


wcss=[]
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)


# In[10]:


wcss


# In[11]:


number_clusters = range(1,10)
plt.plot(number_clusters, wcss)
plt.title('The Elbow Method')
plt.xlabel('No.of clusters')
plt.ylabel('Within-cluster Sum of Squares')


# In[16]:


kmeans = KMeans(3)
kmeans.fit(x_scaled)


# In[13]:


data_with_clusters = data2.copy()
data_with_clusters['Clusters'] = kmeans.fit_predict(x_scaled) 
data_with_clusters.head()


# In[15]:


plt.scatter(data2['Age'],data2['Spending Score (1-100)'], c=data_with_clusters['Clusters'], cmap='rainbow')
plt.title('Spending Score and Age', Fontsize=15)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[17]:


x2 = data2[['Spending Score (1-100)', 'Annual Income (k$)']]
x2_scaled = preprocessing.scale(x2)


# In[18]:


kmeans2 = KMeans(3)
kmeans2.fit(x2_scaled)


# In[19]:


data_with_clusters2 = data2.copy()
data_with_clusters2['Clusters'] = kmeans2.fit_predict(x2_scaled) 
data_with_clusters2


# In[20]:


plt.scatter(data2['Spending Score (1-100)'],data2['Annual Income (k$)'], c=data_with_clusters2['Clusters'], cmap='rainbow')
plt.title('Spending Score and Annual Income', Fontsize=15)
plt.xlabel('Spending Score')
plt.ylabel('Annual Income')
plt.show()


# In[ ]:




