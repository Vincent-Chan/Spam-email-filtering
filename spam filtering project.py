#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all the necessary things

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# read the csv file that contains the email content and the corresponding label (spam/ham)

df = pd.read_csv("email_dataset.csv")


# In[3]:


# visualize the content of the csv file

print(df)


# In[4]:


# show the first 20 rows of content

data = df.where((pd.notnull(df)), "")
data.head(20)


# In[5]:


# let's see the information of the data

data.info()
print("The shape of the csv data is: ", data.shape)


# In[6]:


# if the email is a spam, its label is denoted as 1;
# otherwise, if it is a ham, its label is denoted as 0

data.loc[data["Category"] == "spam", "Category",] = 1
data.loc[data["Category"] == "ham", "Category",] = 0


# In[7]:


# visualize X (Message) and Y (Category)

X = data["Message"]
Y = data["Category"]

print("\nThe emails are as follows: \n")
print(X)
print("\nThe corresponding labels are as follows: \n")
print(Y)


# In[8]:


# split the email data into training dataset and test dataset
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 7)


# In[9]:


# visualize the shape of X, X_train, X_test, Y, Y_train, Y_test

print("The shape of X is: ", X.shape)
print("The shape of X_train is: ", X_train.shape)
print("The shape of X_test is: ", X_test.shape)

print("The shape of Y is: ", Y.shape)
print("The shape of Y_train is: ", Y_train.shape)
print("The shape of Y_test is: ", Y_test.shape)


# In[10]:


# Convert a collection of raw documents to a matrix of TF-IDF features
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = "english", lowercase = True)

# do feature extraction on the train dataset and test dataset
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert the tarin and test label to int type
Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")


# In[11]:


# visualize X_train_features, X_test_features, Y_train, Y_test

print("\nX_train_features: \n", X_train_features)
print("\nX_test_features: \n", X_test_features)
print("\nY_train: \n", Y_train)
print("\nY_test: \n", Y_test)


# In[12]:


# the model used is logistic regression
# and fit the model using X_train_features (dataset) and Y_train (label)

model = LogisticRegression()

model.fit(X_train_features, Y_train)


# In[13]:


# print the accuracy of training dataset

pred_train_data = model.predict(X_train_features)
accu_train_data = accuracy_score(Y_train, pred_train_data)

print("Accuracy on training data: ", accu_train_data)


# In[14]:


# print the accuracy of test dataset

pred_test_data = model.predict(X_test_features)
accu_test_data = accuracy_score(Y_test, pred_test_data)

print("Accuracy on test data: ", accu_test_data)


# In[15]:


# The demo part: copy and paste the email in the textbox
# and it will help to classify whether the email is spam or ham

while True:
    email_input = input("Please copy and paste your email here (enter 'exit' to quite the program): ")
    
    if email_input == "exit":
        print("Program terminates.")
        break
    
    else:
        email_input_list = [email_input]
        input_email_features = feature_extraction.transform(email_input_list)
        prediction = model.predict(input_email_features)
        
        if (prediction[0] == 0):
            print("This email is a ham.")
        else:
            print("This email is a spam.")

