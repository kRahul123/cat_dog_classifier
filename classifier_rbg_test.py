#!/usr/bin/env python
# coding: utf-8

# In[1]:


import classifier_rbg_train as f
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as pilimg
import datetime
import os
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


# In[2]:


def run_for_file(model,path,flag=False):
    img=pilimg.open(path)
    img=img.resize((f.IMG_HEIGHT,f.IMG_WIDTH))
    img=np.divide(img,255)
    img=np.reshape(img,(1,f.IMG_HEIGHT,f.IMG_WIDTH,3))
    x=model.predict(img)
    label=''
    if x[0][0]>0:
        label='Dog'
    else:
        label='Cat'
    if flag:
        plt.title(label)
        plt.imshow(img[0])
        plt.pause(1)
    return label
    


# In[3]:


def run_for_folder(model,path):
    op_file_name='output_'+re.sub("[ :,.]","",str(datetime.datetime.now().time()))+'.txt'
    with open(op_file_name,'w') as x:
        for i in tqdm(os.listdir(path)):
            file_path=os.path.join(path,i)
            label=run_for_file(model,file_path)
            x.write(i+"   :    "+label+'\n')
    print("output written to file "+op_file_name)    
    


# In[4]:


def run_for_accuracy(model,path):
    test_image_generator=ImageDataGenerator(rescale=1./255)
    try:
        test_data_generator=test_image_generator.flow_from_directory(batch_size=f.batch_size,directory=path,shuffle=True,target_size=(f.IMG_HEIGHT,f.IMG_WIDTH),class_mode='binary')
        model.evaluate(test_data_generator)
    except:
        print("Testing data not available!!,  downlaod files from google drive of given link and save in folder \"class_data\"")
	
    
    


# In[5]:


def run():
    [model,history]=f.load_prev_state()
    while True:
        print("Choose option:")
        print("(1) input folder name for test")
        print("(2) input file name for test")
        print("(3) run for accuracy( need to enter properly formatted folder) ")
        print("(4) show training statistics")
        print("(-1) exit ")
        
        option=input()
        if option=='1':
            path=input('Enter folder path\n')
            run_for_folder(model,path)
        elif option=='2':
            path=input('Enter file path\n')
            run_for_file(model,path,True)
        elif option=='3':
            path=input("enter folder path (folder should contain two subfolder named cat and dog)\n")
            run_for_accuracy(model,path)
        elif option=='4':
            f.plot_train_status(history)
        elif option=='-1':
            break
        else:
            print('\n\nInvalid option, choose again!!\n\n')
            
    
    


# In[ ]:





# In[ ]:


if __name__=="__main__":
    run()


# In[ ]:





# In[ ]:




