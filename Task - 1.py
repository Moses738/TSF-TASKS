#!/usr/bin/env python
# coding: utf-8

# # Task-1 Prediction Using Supervised ML

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# 

# In[10]:


df = pd.read_csv(r'C:\Users\Madan Kumar\Desktop\students data - Copy.csv')
df.head()


# In[11]:


plt.scatter(x=df['Hours'], y=df['Scores'])
plt.title('Relation Between Hours And Scores', fontdict={'weight' : 'bold', 'size' : 18})
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[12]:


X = df[['Hours']].values
Y = df[['Scores']].values


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[14]:


model = LinearRegression()
model.fit(x_train, y_train)


# In[15]:


y_predict = model.predict(x_test)
y_predict


# In[23]:


l = (model.coef_ * X) + model.intercept_
plt.scatter(X, Y)
plt.plot(X, l, color='g')
plt.title('Linear regression visualizer', fontdict={'weight' : 'bold', 'size' : 18})
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[24]:


prediction = model.predict([[9.25]])
print('Input in Hours :', 9.25)
print('Predicted output of Score :',prediction[0][0])


# # If a student studies 9.25hr/day then he will score 93.69
