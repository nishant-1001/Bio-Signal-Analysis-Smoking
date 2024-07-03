#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries
#Loading the Data
#Data Cleaning
#One Hot Encoding
#Feature Selection
#Bagging Algorithms


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("smoking.csv")


# In[4]:


df.head()


# In[5]:


df = df.drop(columns=["ID","oral"])
df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


sns.barplot(x = df["gender"],y= df["smoking"])
plt.show()


# In[11]:


#We can visualize that most smokers are male.


# In[12]:


sns.countplot(df["gender"],hue=df["smoking"])


# In[13]:


#We can see the diversity in gender, that in male percentage of smokers are aroung half of the count.


# In[14]:


plt.figure(figsize=(10,5))
df["smoking"].value_counts().plot.pie(autopct = "%0.2f")


# In[15]:


#From the plot we can observed that 36.73% people are smokers.


# In[16]:


plt.figure(figsize = (9,6))
sns.histplot(x = df["age"],hue = df["smoking"])
plt.show()


# In[17]:


#We infer that the number of smokers are mainly from age group 40.


# In[18]:


for i in df.columns:
    if(df[i].dtypes =="int64" or df[i].dtypes == 'float64'):
        sns.boxplot(df[i])
        plt.show()
    


# In[19]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["gender"]=le.fit_transform(df["gender"])
df["tartar"] = le.fit_transform(df["tartar"])
df["dental caries"]= le.fit_transform(df["dental caries"])


# In[20]:


df.head()


# In[21]:


X = df.iloc[:,:-1]
y = df["smoking"]
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
df1 = pd.Series(model.feature_importances_,index = X.columns)
plt.figure(figsize=(8,8))
df1.nlargest(24).plot(kind="barh")
plt.show()


# In[22]:


#Feature Selection is a technique that calculate a score for all the input features for a given model. So out of 24 features we 
#will select the top 15 features based on the score.


# In[23]:


df.info()


# In[24]:


#Logistic Regression


# In[25]:


X = df[["gender","height(cm)","Gtp","hemoglobin","triglyceride","age","weight(kg)","waist(cm)","HDL","serum creatinine","ALT","fasting blood sugar","relaxation","LDL","systolic"]]
y = df["smoking"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state = 42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
accuracy_score(y_test,y_pred)
classification_report(y_test,y_pred)


# In[26]:


#We can infer that logistic regression model has 73% accuracy.


# In[27]:


X = df[["gender","height(cm)","Gtp","hemoglobin","triglyceride","age","weight(kg)","waist(cm)","HDL","serum creatinine","ALT","fasting blood sugar","relaxation","LDL","systolic"]]
y = df["smoking"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state = 42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
accuracy_score(y_test,y_pred)
classification_report(y_test,y_pred)


# In[28]:


#We can infer that Decision Tree Classifier has accuracy score of 78%.

