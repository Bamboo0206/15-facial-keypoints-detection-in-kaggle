# from prepare_data import X_train, y_train, X_test
# 导入数据
import numpy as np
X_train=np.load('X_train_scale.npy')
y_train=np.load('y_train_scale.npy')
X_test=np.load('X_test_scale.npy')

# 以下删除
X_add=np.load('addX.npy')
y_add=np.load('addY.npy')


X_add=X_add.reshape(-1, 96, 96, 1)
X_train=np.vstack([X_train,X_add])
y_train=np.vstack([y_train,y_add])

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=42)  # shuffle train data

print(X_train.shape,y_train.shape)


from export_result import export
from utils import show_result

# create your model here
############################改这里！搭个神经网络
# from keras.layers.advanced_activations import LeakyReLU # ???
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D,MaxPooling2D
from keras.callbacks import EarlyStopping
from  keras.initializers import Constant,RandomUniform

def bulid_model():
    model = Sequential()  # 初始化一个新的网络

    # filters: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
    # kernel_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值。
    # padding: "valid" 或 "same" (大小写敏感)。
    # use_bias: 布尔值，该层是否使用偏置向量。

    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Dropout(0.1))
    # BLOCK 2
    model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Dropout(0.2))

    # BLOCK 1
    model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Dropout(0.3))

    # full connected #私自改了参数 4096个节点pc跑不出来
    model.add(Flatten())
    model.add(Dense(1000, activation='relu',kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=None),bias_initializer=Constant(value=0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu',kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=None),bias_initializer=Constant(value=0.01)))
    model.add(Dense(30)) # 最终输出15个2维坐标

    model.summary()

    model.compile(optimizer='Adam',loss='mean_squared_error')
    return model


model=bulid_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=12, # patience: 没有进步的训练轮数，在这之后训练就会被停止。
                                restore_best_weights=True) #从具有监测数量的最佳值的时期恢复模型权重。如果为 False，则使用在训练的最后一步获得的模型权重。
model.fit(X_train,y_train,epochs = 10000,batch_size = 256,validation_split = 0.1, callbacks=[early_stopping])

ID=''
# 保存模型
model.save(ID+'model.h5')
# 预测
pred_y = model.predict(X_test,verbose=1)
pred_y = pred_y * 48 + 48 # 缩放
export(pred_y, ID+'2017211444.csv') # use complete student id as the export filename
show_result(X_test[0].reshape(96,96), pred_y[0]) #画图