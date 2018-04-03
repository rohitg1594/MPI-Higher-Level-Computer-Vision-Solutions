
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
from utils import load_data

def showdigit(x,):
    "Show one digit at a time"
    return plt.imshow(x.reshape(28,28), norm=mpl.colors.Normalize(0,255), cmap='gray')

def savedigit(x, file):
    "Save one digit as a gray-scale image"
    norm = mpl.colors.Normalize(0,255)
    plt.imsave(fiel,norm(x.reshape(28,28), cmap='gray'))

def showdigits(X, y, max_digits=15):
    "Show up to max_digits random digits per class from X with class labels from y."
    num_cols = min(max_digits,  max(Counter(y).values()))
    for c in range(10):
        ii = np.where(y==c)[0]
        if len(ii)>max_digits:
            ii = np.random.choice(ii, size=max_digits, replace=False)
        for j in range(num_cols):
            ax = plt.gcf().add_subplot(10, num_cols, c*num_cols+j+1, aspect='equal')
            ax.get_xaxis().set_visible(False)
            if j==0:
                ax.set_ylabel(c)
                ax.set_yticks([])
            else:
                ax.get_yaxis().set_visible(False)
            if j<len(ii):
                ax.imshow(X[ii[j],].reshape(28,28), norm=mpl.colors.Normalize(0,255), cmap='gray')
            else:
                ax.axis('off')


