#!/usr/bin/env python
# coding: utf-8

# # Boston Housing Dataset for Regression Machine Learning

# # Task

# A Boston House Dataset by sklearn will be used for our prediction. The goal is to be able to make a price prediction of a house and to determine the factors on which the price depends.

# # Import Libraries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Boston Housing Dataset from the Scikit Learn Repository 

# In[2]:


from sklearn.datasets import load_boston
boston = load_boston()


# # Summarize the Dataset 

# Taking a look at the data in the following ways:

# Peek at the Data Quality Check Statistical Summary of all Attributes

# # Peek at the data

# In[3]:


#Print out a description of the dataset.
print(boston.DESCR)


# In[4]:


#Take a look at the column names.
boston.feature_names


# In[5]:


#Convert the dataset into a pandas dataframe.
df = pd.DataFrame(boston.data, columns = boston.feature_names)


# In[6]:


# Take a look at the first 5 rows of the data.
df.head()


# House prices (MEDV) did not appear in the dataframe. They are the target of the boston dataframe and therefore, needs to be added to the dataframe.

# In[7]:


df['MEDV'] = boston.target


# In[8]:


df.head()


# # Quality Check

# This is to check if there are null-values in the dataset

# In[9]:


df.info()


# # Statistical summary of all attributes

# In[10]:


df.describe()


# # Visualize the Data

# Making a quick visualization for the data. This is to:

# Understand the distribution of each feature.
# 
# Find the correlation between the features.
# 
# Identify the features that correlates most win the House price (MEDV)
# 

# In[11]:


sns.pairplot(df)


# # Distribution

# In[12]:


rows = 7 
cols = 2
fig, ax = plt.subplots(nrows= rows, ncols= cols, figsize= (16,16))
col = df.columns
index = 0
for i in range(rows):
    for j in range(cols):
        sns.distplot(df[col[index]], ax = ax[i][j])
plt.tight_layout()


# # Correlation

# Summarize the relationships between the variables 

# In[13]:


fig, ax= plt.subplots(figsize = (16,9))
sns.heatmap(df.corr(), annot = True, annot_kws={'size':12})


# From the correlation matrix, in regard to our target column, it is observed that the house pricing(MEDV) has a strong postive relationship with RM and strong neagtive relationship with LSTAT.

# -It also has moderate postive/negative relationship with the other columns

# -For a linear regression method, a nearly high correlation is needed hence a threshold filter must be defined

# -For this reason, we need to define a function called getCorrelatedFeature

# In[14]:


def getCorrelatedFeature(corrdata, threshold):
    feature = []
    value = []
    
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index])>threshold:
            feature.append(index)
            value.append(corrdata[index])
            
    df = pd.DataFrame(data = value, index = feature, columns = ['Corr Value'])
    return df


# In[15]:


#Setting a threshold limit of 0.4.
threshold = 0.4
corr_value = getCorrelatedFeature(df.corr()['MEDV'], threshold)


# In[16]:


#Checking out the dependencies after applying the threshold limit
corr_value.index.values


# # Quick view of correated data

# In[17]:


correlated_data = df[corr_value.index]
correlated_data.head()


# # Linear Regression

# # Split and Test dataset

# In[18]:


X = correlated_data.drop(labels = ['MEDV'], axis =1)
y = correlated_data['MEDV']


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 1)


# In[20]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[21]:


lm.fit(X_train, y_train)


# # Results visualization

# The goal is to have a perfect linear or nearly linear relation of the points. The larger the distribution of the points, the greater the inaccuracy of the model

# In[22]:


#Predict with the X_test data
predictions = lm.predict(X_test)


# In[23]:


#Plot the y-test against the predicted values on a scatter plot
plt.scatter(y_test, predictions)


# In[24]:


sns.distplot((y_test-predictions),bins = 50)


# # The y-axis of a linear function

# In[25]:


lm.intercept_


# # Coefficients of a linear regression function

# In[26]:


lm.coef_


# # Linear regression function

# In[27]:


#Define linear regression function.
def lin_func(values, coefficients= lm.coef_, y_axis = lm.intercept_):
    return np.dot(values, coefficients) +y_axis


# # Predicition Samples

# In[28]:


from random import randint
for i in range(5):
    index = randint(0, len(df)-1)
    sample = df.iloc[index][corr_value.index.values].drop('MEDV')
    print('PREDICTION: ', round(lin_func(sample),2),
         '// REAL: ', df.iloc[index]['MEDV'],
         '//DIFFERENCE: ', round(round(lin_func(sample),2) - df.iloc[index]['MEDV'],2))


# # The dependencies in the model are:

# INDUS, NOX, RM, TAX, PTRATION and LSTAT

# Because they havE THE |correlation coefficients|>0.4 which is the threshold value.
