#!/usr/bin/env python
# coding: utf-8

# # Insurance_Linear_Regression

# In[ ]:


Author ~ Saurabh 
Date ~ 04-Dec-21


# In[1]:


#importing lib
import os
import numpy as np
import pandas as pd

#for ploting 
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
sns.set_style("darkgrid")
import ipywidgets as widgets
from IPython.display import display

#to supress warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#pandas profiling
import pandas_profiling as pp


# In[3]:


pwd


# In[4]:


#listdir
os.listdir("E:\\DataScience\\MachineLearning\\Insurance_Linear_Regression")


# In[5]:


#read dataset..
df =pd.read_csv("E:\\DataScience\\MachineLearning\\Insurance_Linear_Regression\\Insurance_Data.csv")


# In[6]:


profile =df.profile_report(title="Insurance Data Profile Report")


# In[7]:


profile


# In[8]:


#profile report
profile.to_file(output_file="Insurance_profile_report.html")


# In[9]:


#top 5 columns of data
df.head()


# In[10]:


#info
df.info()


# In[11]:


#shape
df.shape


# In[12]:


#tail of dataframe
df.tail()


# In[13]:


#count of null values
df.isnull().sum()


# In[14]:


#unique value per column
df.nunique()


# In[15]:


# numerical data
df.corr()


# In[33]:


#categorical data
catogorical= []
lst =df.columns
for i in range(7):
    if(df[lst[i]].dtype =='object'):
        catogorical.append(lst[i])
        
print("catogorical columns :",catogorical)


# In[36]:


#encoding
from sklearn.preprocessing import LabelEncoder
lb_encoder=LabelEncoder()


# In[40]:


df['sex'] =lb_encoder.fit_transform(df['sex'])
df['smoker']=lb_encoder.fit_transform(df['smoker'])
df['region'] =lb_encoder.fit_transform(df['region'])


# In[41]:


df.head()


# In[52]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),cbar=True,square=True,annot=True,annot_kws={'size':10},fmt='.2f',cmap='Dark2_r')


# In[44]:


#input and target
X =df.iloc[:,:-1]
y =df.iloc[:,-1]


# In[45]:


#shape
print("Shape of Input :",X.shape)
print("Shape of Target:",y.shape)


# In[47]:


#train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=11)


# In[49]:


#shape
print("Shape X_train :",X_train.shape)
print("Shape y_train :",y_train.shape)
print("Shape X_test :",X_test.shape)
print("Shape y_test :",y_test.shape)


# In[53]:


#scaling the given data
from sklearn.preprocessing import StandardScaler
Scaler =StandardScaler()
X_train_scale =Scaler.fit_transform(X_train)
X_test_scale =Scaler.fit_transform(X_test)


# # Linear Regression

# In[54]:


#importing linear regession
from sklearn.linear_model import LinearRegression


# In[56]:


#fiiting the data to model
model =LinearRegression()
model.fit(X_train,y_train)


# In[57]:


#y_pred : predicted values
y_pred=model.predict(X_test)


# In[58]:


#Report of model
print('Coefficients: \n', model.coef_)
print("Mean squared error: %.2f" % np.mean((model.predict(X_test) - y_test) ** 2))
print('Variance score: %.2f' % model.score(X_test, y_test))


# In[59]:


def accuracy(X_test,y_test, y_pred):
    print('accuracy (R^2):\n', model.score(X_test, y_test)*100, '%')


# In[60]:


accuracy(X_test, y_test, y_pred)


# # Xgboost

# In[63]:


#training with Xgboost regressor
import xgboost as xgb
modelX = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.2, max_depth = 10, n_estimators = 100)
modelX.fit(X_train, y_train)


# In[64]:


y_hat =modelX.predict(X_test)


# In[67]:


result = modelX.score(X_test, y_test)
print("Accuracy : {}".format(result))


# # RandomForestRegressor
# 

# In[68]:


from sklearn.ensemble import RandomForestRegressor
modelR=RandomForestRegressor()
modelR.fit(X_train,y_train)


# In[69]:


y1_hat =modelR.predict(X_test)


# In[70]:


result = modelR.score(X_test, y_test)
print("Accuracy : {}".format(result))


# # Interactive Display Widgets

# In[75]:


age_widget = widgets.IntSlider(
   value=38,
   min=18,
   max=64,
   step=1,
   description="Age:"
)

bmi_widget = widgets.FloatSlider(
   value=30,
   min=15,
   max=54,
   step=0.01,
   description="BMI:"
)

children_widget = widgets.IntSlider(
   value=1,
   min=0,
   max=5,
   step=1,
   description="Children:"
)

sex_widget = widgets.ToggleButtons(
   options=[('female',0),('Male',1)],
   description="Sex:"
)

smoker_widget = widgets.ToggleButtons(
   options=[('no',0),('yes',1)],
   description="Smoker:"
)

region_widget = widgets.Dropdown(
   options=[('northeast',4),('Southeast',2),('Northwest',3),('Southwest',1)],
   description="Region:"
)

predict_btn = widgets.Button(
   description="Predict"
)

prediction_out = widgets.Output()


def make_prediction(btn):
   x = pd.DataFrame({
       'age':      age_widget.value,
       'sex':      sex_widget.value,
       'bmi':      bmi_widget.value,
       'children': children_widget.value,
       'smoker':   smoker_widget.value,
       'region':   region_widget.value
   }, index=[0])
   
   prediction = modelR.predict(x)
   
   with prediction_out:
       prediction_out.clear_output()
       print("Prediction: {:.4f}".format(prediction[0]))


predict_btn.on_click(make_prediction)


display(age_widget, bmi_widget, children_widget, sex_widget, smoker_widget, region_widget, predict_btn, prediction_out)


# In[ ]:




