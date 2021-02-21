import matplotlib.pyplot as plt
import numpy as np

def show_result(image, points):
    plt.imshow(image, cmap='gray')
    for i in range(15):
        plt.plot(points[2*i], points[2*i + 1], 'ro')
    plt.show()

def show_16_result(x,y):
    x.reshape(-1,96,96)
    for k in range(16):
        plt.subplot(4,4,k+1) #4*4子图，占第k+1
        plt.imshow(x[k].reshape(96,96), cmap='gray')
        for i in range(15):
            plt.plot(y[k][2 * i], y[k][2 * i + 1], 'ro')
    plt.show()