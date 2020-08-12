#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[15]:


dataset = pd.read_csv('First set CSV.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values


# In[16]:


from sklearn.ensemble import RandomForestRegressor 


# In[17]:


regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X,y)


# In[18]:


y_pred = regressor.predict(X)


# In[19]:


import pickle


# In[20]:


file=open('model.pkl','wb')


# In[21]:


pickle.dump(regressor, file)


# In[ ]:




