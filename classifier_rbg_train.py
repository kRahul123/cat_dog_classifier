#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


train_dir='class_data/train'
validation_dir='class_data/validation'
test_dir='class_data/test'


# In[3]:


batch_size=50
epochs=50
IMG_HEIGHT=125
IMG_WIDTH=125


# In[4]:


train_image_generator=ImageDataGenerator(rescale=1./255,rotation_range=45,width_shift_range=.15,height_shift_range=.15,horizontal_flip=True,zoom_range=0.5)
validation_image_generator=ImageDataGenerator(rescale=1./255)
test_image_generator=ImageDataGenerator(rescale=1./255)


# In[5]:


## Create data generator



# In[6]:


def create_model():
    model=Sequential([
    Conv2D(16,5,padding='same',activation='relu',input_shape=(IMG_HEIGHT,IMG_WIDTH,3)),
    MaxPooling2D(),
    Conv2D(32,5,padding='same',activation='relu'),
    MaxPooling2D(),
    Conv2D(64,5,padding='same',activation='relu'),
    MaxPooling2D(),
    Conv2D(128,5,padding='same',activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512,activation='relu'),
    Dense(1)
    
    ])

    model.compile(optimizer='adam',
             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
             metrics=['accuracy'])
    model.summary()
    return model


# In[7]:



def train_model(model,train_data_gen,validation_data_gen):
    history=model.fit(
        train_data_gen,
        steps_per_epoch=int(15000/batch_size),
        epochs=epochs,
        validation_data=validation_data_gen,
        validation_steps=int(5000/batch_size)
    )
    return [model,history]


# In[8]:


def save_state(history,model):
    model.save('cat_dog_model/MyModel',save_format='tf')
    import pickle
    with open('cat_dog_model/historyDict','wb') as file_pi:
        pickle.dump(history.history,file_pi)


# In[9]:


def plot_train_status(history):
    acc = history['accuracy']
    val_acc =history['val_accuracy']

    loss=history['loss']
    val_loss=history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    


# In[10]:


def load_prev_state():
    model=tf.keras.models.load_model('cat_dog_model/MyModel')
    import pickle
    history=pickle.load(open('cat_dog_model/historyDict','rb'))
    return [model,history]


# In[11]:


def plot_with_tag(img_arr):
    for img in img_arr:
        pred=model.predict(np.reshape(img,(1,100,100,3)))
        plt.title('dog') if pred[0][0]>0 else plt.title('cat')
        plt.imshow(img)
        plt.pause(0.5)
       
        
    


# In[ ]:





# In[ ]:


if __name__=="__main__":
    train_data_gen=train_image_generator.flow_from_directory(batch_size=batch_size,directory=train_dir,shuffle=True,target_size=(IMG_HEIGHT,IMG_WIDTH),class_mode='binary')
    validation_data_gen=validation_image_generator.flow_from_directory(batch_size=batch_size,directory=validation_dir,shuffle=True,target_size=(IMG_HEIGHT,IMG_WIDTH),class_mode='binary')
    model=create_model()
    [model,history]=train_model(model,train_data_gen,validation_data_gen)
    save_state(history,model)


# In[ ]:





# In[ ]:





# In[ ]:




