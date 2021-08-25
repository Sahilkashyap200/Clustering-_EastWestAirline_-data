#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[2]:


airlines_=pd.read_csv("\\Users\\piyus\\Documents\\EastWestAirline.csv")


# In[3]:


airlines_.head()


# In[4]:


airlines_.shape


# In[5]:


airlines=airlines_.iloc[:,1:]


# In[6]:


airlines.dtypes


# In[7]:


airlines.isna().sum()


# In[8]:


#scaling the data
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
scale.fit(airlines.iloc[:,0:])
airlines.iloc[:,0:]=scale.transform(airlines.iloc[:,0:])
airlines.head()


# In[9]:


#elbow method to decide K 
k_rng=range(1,11)
sse= []
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(airlines)
    sse.append(km.inertia_)


# In[10]:


sse


# In[11]:


from matplotlib import pyplot as plt
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(k_rng,sse)


# # KMeans Clustering 

# In[12]:


model=KMeans(n_clusters=2)
pred=model.fit_predict(airlines)
pred


# In[13]:


airlines_=airlines_.drop('ID#',axis=1)


# In[14]:


airlines_['Cluster']=pred


# In[15]:


airlines_.head()


# In[16]:


airlines_.groupby(airlines_.Cluster).mean()


# # DBSCAN

# In[17]:


model_=DBSCAN(eps=0.3,min_samples=13)
pred_=model_.fit_predict(airlines_)
pred_


# In[18]:


airlines_['clust']=pred_


# In[19]:


airline=airlines_.drop('Cluster',axis=1)


# In[20]:


airline.head()


# In[ ]:





# In[21]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[22]:


#creating dendrogram
dendrogram=sch.dendrogram(sch.linkage(airlines_,method='single'))


# In[23]:


HC=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='single')
y_hc=HC.fit_predict(airlines_)
clusters=pd.DataFrame(y_hc,columns=['Clusters'])
clusters


# In[ ]:




