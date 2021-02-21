#!/usr/bin/env python
# coding: utf-8

# # 数据预处理
# ***
# 读取数据

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# 从文件中读取数据
Train_Dir = './data/train.csv'
Test_Dir = './data/test.csv'
train_data = pd.read_csv(Train_Dir)
test_data = pd.read_csv(Test_Dir)



# 数据处理：缺失值填充，标签与数据分类

# In[8]:


# use the previous value to fill the missing value
train_data.fillna(method='ffill', inplace=True)

# preparing training data
imga = []
for i in range(len(train_data)):
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    imga.append(img)

image_list = np.array(imga, dtype='float')
X_train = image_list.reshape(-1, 96, 96, 1)

# preparing training label
training = train_data.drop('Image', axis=1)
y_train = []
for i in range(len(train_data)):
    y = training.iloc[i, 1:]
    y_train.append(y)
y_train = np.array(y_train, dtype='float64')

# preparing test data
timga = []
for i in range(len(test_data)):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]
    timga.append(timg)
timage_list = np.array(timga, dtype='float64')
X_test = timage_list.reshape(-1, 96, 96, 1)


# In[9]:


X_train.shape,y_train.shape,X_test.shape


# # 数据增强
# ***
# 将图像翻转，获得增倍的数据集

# In[10]:


index=[i for i in range(96)]
index.reverse() #图片翻转：获得需要交换的列的索引


# In[11]:


# 将原来的左右标签交换（例如left eye center 与 right eye center 交换），交换的列如下
y_i=[2,
 3,
 0,
 1,
 8,
 9,
 10,
 11,
 4,
 5,
 6,
 7,
 16,
 17,
 18,
 19,
 12,
 13,
 14,
 15,
 20,
 21,
 24,
 25,
 22,
 23,
 26,
 27,
 28,
 29]


# In[12]:


# 反转图片，数据倍增，图像增强
X_add=X_train[:,:,index]
y_add=y_train[:,y_i]


# 将坐标做轴对称变换

# In[13]:


tmp=[-95,0]*15
tmp=[tmp]*6001
tmp
y_add=y_add+tmp


# In[14]:


from math import fabs
y_add=abs(y_add)
# y_add


# In[15]:


# 结果
y_add[0],y_train[0]


# In[16]:


X_add.shape,y_add.shape







# In[21]:


# 结果拼接
X_train=np.vstack([X_train,X_add])
y_train=np.vstack([y_train,y_add])
X_train.shape,y_train.shape


# In[22]:


# X_train.shape,y_train.shape


# 数据规范化

# In[23]:


# 缩放：将图像像素的强度值缩放为[0，1]区间，而不是0到255。目标值（x和y坐标）缩放为[- 1，1];在0到95之间。
X_train=X_train/255
X_test=X_test/255
y_train=(y_train-48)/48


# In[24]:


#打乱数据顺序
from sklearn.utils import shuffle
X_train,y_train=shuffle(X_train,y_train)


