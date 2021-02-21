# from DataPreprocessing import X_test
from prepare_data import X_test
from keras.models import load_model
from export_result import export
from utils import show_result,show_16_result
# from model import bulid_model

# model=bulid_model()
# model.load_weights('model.h5')
model=load_model('model.hdf5')

# 预测
pred_y = model.predict(X_test,verbose=1)
pred_y = pred_y * 48 + 48 # 缩放
export(pred_y, '2017211444.csv') # use complete student id as the export filename
show_16_result(X_test[0:16].reshape(-1,96,96),pred_y[0:16])
show_result(X_test[0].reshape(96,96), pred_y[0]) #画图