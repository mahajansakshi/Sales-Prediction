#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
sns.set(style="darkgrid")


# In[4]:


data=pd.read_csv('Advertising.csv')
data.head()


# ## EDA :- 

# In[5]:


data.drop('Unnamed: 0', axis=1, inplace=True)
data.shape


# In[6]:


data.isna().sum()


# In[7]:


data.describe()


# In[8]:


data.duplicated().sum()


# In[9]:


data.info()


# ## Data Visualisation :- 

# In[10]:


plt.figure(figsize=(15,8))
for i,col in enumerate(['TV','Radio','Newspaper']):
    plt.subplot(2,2,i+1)
    sns.regplot(data=data, x=col, y='Sales')


# In[11]:


plt.figure(figsize=(15,8))
for i,col in enumerate(['TV','Radio','Newspaper']):
    plt.subplot(2,2,i+1)
    sns.histplot(data=data,x=col,bins=20,kde=True)


# In[12]:


plt.figure(figsize=(15,8))
for i,col in enumerate(['TV','Radio','Newspaper','Sales']):
    plt.subplot(2,2,i+1)
    sns.boxplot(data=data, x=col, orient='h')


# In[13]:


data.corr()


# In[14]:


sns.heatmap(data.corr(), annot=True)


# In[15]:


data['Total_advertising']=data['TV']+data['Radio']+data['Newspaper']
data.head()


# In[16]:


data.corr()


# ## Scaling & Spliting of Data :- 

# In[17]:


X=(data.drop(columns=['Sales'])).values
Y=data[['Sales']].values.flatten()
X[:5,], Y[:5]


# In[18]:


x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
x_train.shape, x_test.shape


# In[19]:


scaler=StandardScaler()
x_train_scale=scaler.fit_transform(x_train)
x_test_scale=scaler.fit_transform(x_test)
x_train_scale[:5,]


# ## Model Training :- 

# ### Linear regression 

# In[20]:


lin_model=LinearRegression()
lin_model.fit(x_train_scale, y_train)

y_train_pred=lin_model.predict(x_train_scale)
y_test_pred=lin_model.predict(x_test_scale)

mse_train=mean_squared_error(y_train, y_train_pred)
mse_test=mean_squared_error(y_test, y_test_pred)
r2_train=r2_score(y_train, y_train_pred)
r2_test=r2_score(y_test, y_test_pred)

print('Linear Regression Evaluation =>\n\tTrain:\n\t\tMSE: {}\n\t\tR2 Score: {}\n\tTest:\n\t\tMSE: {}\n\t\tR2 Score: {}'.format(mse_train, r2_train,mse_test,r2_test))


# ### Random forest regressor 

# In[21]:


rfr_model=RandomForestRegressor(random_state=42)
rfr_model.fit(x_train_scale, y_train)

y_train_pred=rfr_model.predict(x_train_scale)
y_test_pred=rfr_model.predict(x_test_scale)

mse_train=mean_squared_error(y_train, y_train_pred)
mse_test=mean_squared_error(y_test, y_test_pred)
r2_train=r2_score(y_train, y_train_pred)
r2_test=r2_score(y_test, y_test_pred)

print('Random Forest Regressor Evaluation =>\n\tTrain:\n\t\tMSE: {}\n\t\tR2 Score: {}\n\tTest:\n\t\tMSE: {}\n\t\tR2 Score: {}'.format(mse_train, r2_train,mse_test,r2_test))


# ### Gradient boosting regressor 

# In[22]:


gbr_model=GradientBoostingRegressor(random_state=42,loss='squared_error')
gbr_model.fit(x_train_scale, y_train)

y_train_pred=gbr_model.predict(x_train_scale)
y_test_pred=gbr_model.predict(x_test_scale)

mse_train=mean_squared_error(y_train, y_train_pred)
mse_test=mean_squared_error(y_test, y_test_pred)
r2_train=r2_score(y_train, y_train_pred)
r2_test=r2_score(y_test, y_test_pred)

print('Gradient Boosting Regressor Evaluation =>\n\tTrain:\n\t\tMSE: {}\n\t\tR2 Score: {}\n\tTest:\n\t\tMSE: {}\n\t\tR2 Score: {}'.format(mse_train, r2_train,mse_test,r2_test))


# ## Model Result Analysis :-

# In[23]:


plt.figure(figsize=(14,4))
sns.regplot(x=y_test,y=y_test_pred,color='purple')
plt.title('Actual vs Predicted Selling Price (Gradient Boosting Regressor)')
plt.xlabel('Actual Test Values')
plt.ylabel('Predicted Test Values')


# In[ ]:




