
# 导入数据
from DataPreprocessing import X_train,y_train,X_test
print(X_train.shape,y_train.shape)

from export_result import export
from utils import show_result,show_16_result

# create your model here
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
from keras.callbacks import EarlyStopping, ModelCheckpoint


def bulid_model():
    model = Sequential()  # 初始化一个新的网络

    # filters: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
    # kernel_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值。
    # padding: "valid" 或 "same" (大小写敏感)。
    # use_bias: 布尔值，该层是否使用偏置向量。

    # BLOCK 1
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False, input_shape=(96, 96, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())  # 在每一个批次的数据中标准化前一层的激活项， 即，应用一个维持激活项平均值接近 0，标准差接近 1 的转换。https://keras.io/zh/layers/normalization/#batchnormalization

    model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # BLOCK 2
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # BLOCK 3
    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(96, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # BLOCK 4
    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # BLOCK 5
    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # BLOCK 6
    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    # full connected
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(30))
    model.summary()


    model.compile(optimizer='adam',loss='mean_squared_error')
    return model


model=bulid_model()

# 保存最好模型
ID=''
filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor = "val_loss",verbose = 1,save_best_only = "True",mode = "auto",period = 1)
callbacks_list= [checkpoint]
# early_stopping = EarlyStopping(monitor='val_loss', patience=12, # patience: 没有进步的训练轮数，在这之后训练就会被停止。
#                                 restore_best_weights=True,verbose=1) #从具有监测数量的最佳值的时期恢复模型权重。如果为 False，则使用在训练的最后一步获得的模型权重。
model.fit(X_train,y_train,epochs = 50,batch_size = 256,validation_split = 0.1, callbacks=callbacks_list)


# 保存模型
model.save(ID+'model.h5')
# 预测
pred_y = model.predict(X_test,verbose=1)
pred_y = pred_y * 48 + 48 # 缩放
export(pred_y, ID+'2017211444.csv') # use complete student id as the export filename
# show_result(X_test[0].reshape(96,96), pred_y[0]) #画图
show_16_result(X_test[0:16].reshape(-1,96,96),pred_y[0:16])