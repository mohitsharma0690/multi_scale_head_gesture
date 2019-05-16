import numpy as np
import scipy as sp
from scipy.interpolate import splev, splrep
from scipy.interpolate.rbf import Rbf
import matplotlib.pyplot as plt
import seaborn as sns

def main_2():
    x = np.linspace(0, 16, 16)
    # y = np.random.rand(16)
    y = np.sin(x)
    tck = splrep(x, y, k=3)
    x2 = np.linspace(0, 16, 48)
    y2 = splev(x2, tck)
    plt.subplot(211)
    plt.plot(np.linspace(0, 48, 48), y2)
    plt.subplot(212)
    plt.plot(x, y)
    plt.show()
    return
    # y2 has shape 32

    new_y2 = y2[12:48]
    y3 = new_y2[np.arange(0, 31, 2).astype(int)]
    y4 = new_y2[np.arange(1, 32, 2).astype(int)]

    plt.plot(x, y, 'r-', x, y3, 'g--', x, y4, 'b:')
    #plt.plot(x2,y2)
    plt.show()

def main_3():
    x = np.linspace(0, 16, 16)
    # y = np.random.rand(16)
    y = np.sin(x)
    tck = splrep(x, y, k=3)
    x2 = np.linspace(0, 16, 32)
    y2 = splev(x2, tck)
    # y2 has shape 32

    new_y2 = y2[0:0+16]
    y3 = new_y2[np.arange(0, 16, 1).astype(int)]

    plt.plot(x, y, 'r-', x, y3, 'g--')
    #plt.plot(x2,y2)
    plt.show()

def main_compress():
    x = np.linspace(1, 32, (32*5)/4)
    y = np.random.rand((32*5)/4)
    #y = np.sin(x)
    tck = splrep(x, y, k=3)
    rbf_adj = Rbf(x, y, function='gaussian')
    x2 = np.linspace(2, 33, 32)
    y2 = splev(x2, tck)
    y2_rbf = rbf_adj(x2)
    print(y2)
    print(y2_rbf)
    # y2 has shape 32

    plt.subplot(311)
    plt.plot(x, y)
    plt.subplot(312)
    #plt.plot(x2, y[8:8+16], 'r-', x2, y2, 'g--')
    plt.plot(x2,y2,'go-')
    #plt.plot(x2,y2)
    plt.subplot(313)
    plt.plot(x2, y2_rbf, 'ro-')
    plt.show()

def main_expand():
    x = np.linspace(1, 16, 16)
    y = np.random.rand(16)
    #y = np.sin(x)
    tck = splrep(x, y, k=3)
    x2 = np.linspace(1, 16, 32)
    y2 = splev(x2, tck)
    # y2 has shape 32

    #new_y2 = y2[8:8+32]
    #y3 = new_y2[np.arange(0, 31, 2).astype(int)]
    #y4 = new_y2[np.arange(1, 32, 2).astype(int)]
    plt.subplot(211)
    plt.plot(x, y)
    plt.subplot(212)
    plt.plot(np.linspace(1,32,32), y2)

    #plt.plot(x, y, 'r-', x, y3, 'g--', x, y4, 'b:')
    #plt.plot(x2,y2)
    plt.show()

def main():
    x = np.linspace(0, 16, 16)
    # y = np.random.rand(16)
    y = np.sin(x)
    tck = splrep(x, y, k=3)
    x2 = np.linspace(0, 16, 48)
    y2 = splev(x2, tck)
    # y2 has shape 32

    new_y2 = y2[8:8+32]
    y3 = new_y2[np.arange(0, 31, 2).astype(int)]
    y4 = new_y2[np.arange(1, 32, 2).astype(int)]

    plt.plot(x, y, 'r-', x, y3, 'g--', x, y4, 'b:')
    #plt.plot(x2,y2)
    plt.show()

if __name__ == '__main__':
    main_compress()
    #main_expand()
