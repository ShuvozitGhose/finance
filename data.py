import numpy as np
import pandas as pd
import os, datetime
from pandas_datareader import data as pdr

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
print('Tensorflow version: {}'.format(tf.__version__))
import pdb
#import yfinance as yf

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')


# download dataframe
data = pdr.get_data_yahoo("^GSPC", start="2005-01-01", end="2016-01-01")


data.to_csv("stock.csv")