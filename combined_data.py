import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests as r
from math import sin, cos, sqrt, atan2, radians
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SequentialFeatureSelector, RFECV

random_state=42
num_samples = 75000
itr = 50

file1 = 'https://envera-consulting-public-assets.s3.us-west-1.amazonaws.com/ssie-637-sample-2022-11-26-25k_random_state_42.csv'
file2 = 'https://envera-consulting-public-assets.s3.us-west-1.amazonaws.com/ssie-637-sample-2022-11-26-25k-part-2_random_state_1.csv'
file3 = 'https://envera-consulting-public-assets.s3.us-west-1.amazonaws.com/ssie-637-sample-2022-11-26-25k-part-2_random_state_95.csv'
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

df = pd.concat([df1, df2, df3])

df.drop_duplicates(subset='EMS Call Number', keep='first', inplace=True)
df.to_csv('ssie-637-final-dataset.csv', index=False)
