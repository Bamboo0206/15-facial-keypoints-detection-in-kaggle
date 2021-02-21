# from prepare_data import X_train, y_train, X_test
# 导入数据
import numpy as np
X_train=np.load('X_train.npy')
y_train=np.load('y_train.npy')
X_test=np.load('X_test.npy')

from export_result import export
from utils import show_result

# create your model here
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping
from  keras.initializers import RandomNormal,Constant

def bulid_model():
    model = Sequential()  # 初始化一个新的网络

    # filters: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
    # kernel_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值。
    # padding: "valid" 或 "same" (大小写敏感)。
    # use_bias: 布尔值，该层是否使用偏置向量。

    # BLOCK 1
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(96, 96, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # BLOCK 2
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # BLOCK 3
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu',padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # BLOCK 4
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu',padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # BLOCK 5
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu',padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # full connected #私自改了参数 4096个节点pc跑不出来
    model.add(Flatten())
    model.add(Dense(512, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.005, seed=None),bias_initializer=Constant(value=0.1)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.005, seed=None),bias_initializer=Constant(value=0.1)))
    model.add(Dropout(0.1))
    model.add(Dense(30, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.005, seed=None),bias_initializer=Constant(value=0.1))) # 最终输出15个2维坐标
    model.summary()


    model.compile(optimizer='sgd',loss='mean_squared_error')
    return model


model=bulid_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=20, # patience: 没有进步的训练轮数，在这之后训练就会被停止。
                                restore_best_weights=True) #从具有监测数量的最佳值的时期恢复模型权重。如果为 False，则使用在训练的最后一步获得的模型权重。
model.fit(X_train,y_train,epochs = 50,batch_size = 256,validation_split = 0.1, callbacks=[early_stopping])

ID=''
# 保存模型
model.save(ID+'model.h5')
# 预测
pred_y = model.predict(X_test,verbose=1)
export(pred_y, ID+'2017211444.csv') # use complete student id as the export filename
show_result(X_test[0].reshape(96,96), pred_y[0]) #画图