import numpy as np

#######################需要换成rmse？
def calculate_mse(predict, label):
    mse_array = ((predict - label)**2).mean(axis=0)
    return mse_array.mean()
