#!/usr/bin/env python
# coding: utf-8

# In[78]:


#importing 
from keras.datasets import mnist


# In[79]:


import keras


# In[3]:


from keras.models import Sequential


# In[4]:


from keras.utils.np_utils import to_categorical


# In[5]:


from keras.layers import Dense


# In[6]:


from keras.optimizers import Adam


# In[7]:


from keras.backend import clear_session


# In[8]:


#loading model
(X_train , y_train), (X_test , y_test) = mnist.load_data("mymnist.db")


# In[9]:


#reshaping the data
X_test_ = X_test.reshape(-1 , 28*28)
X_train = X_train.reshape(-1 ,  28*28)
# changing the datatype
X_test = X_test.astype("float32")
X_train = X_train.astype("float32")


# In[10]:


#performing One hot encoding
y_test= to_categorical(y_test)


# In[11]:


y_train= to_categorical(y_train)


# In[67]:


#creating model and layers
model = Sequential()


# In[68]:


model.add(Dense(units = 20, input_dim = 28*28 , activation = 'relu'))


# In[69]:


model.add(Dense(units=120, input_dim = 28*28 , activation = 'relu'))


# In[70]:


model.add(Dense(units=50, input_dim = 28*28 , activation = 'relu'))


# In[71]:


model.add(Dense(units=10, input_dim = 28*28 , activation = 'softmax'))


# In[72]:


model.compile( optimizer= "Adam" , loss='categorical_crossentropy', 
             metrics=['accuracy'] )


# In[73]:


model.summary()


# In[74]:


model_history = model.fit(X_train,y_train,epochs=3,verbose=False)


# In[75]:


text = model_history.history


# In[1]:


accuracy = text['accuracy'][2] * 100


# In[77]:


accuracy = int(accuracy)
f= open("accuracy.txt","w+")
f.write(str(accuracy))
f.close()
print("Accuracy for the model is : " , accuracy ,"%")


# In[ ]:




