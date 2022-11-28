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

random_state=95
num_samples = 25000
itr = 500

# Register for an API key at `positionstack.com` and enter the key that you get below.
access_key = '<INSERT API KEY HERE>'

url = 'https://s3.amazonaws.com/vbgov-ckan-open-data/EMS+Calls+For+Service.csv'
df = pd.read_csv(url)
newdf = df.dropna(how = 'any')

new_df = newdf.copy().sample(num_samples, random_state=random_state).reset_index(drop=True)
new_df['address_query_string'] = (new_df['Block Address'] + ' ' + new_df['City'] + ' ' + new_df['State']).str.lower()

# Formatting the date-time strings to true date-time objects.
date_string = '%Y%m%d %H:%M:%S'
new_df['call_date_and_time'] = pd.to_datetime(new_df['Call Date and Time'], format=date_string)
new_df['entry_date_and_time'] = pd.to_datetime(new_df['Entry Date and Time'], format=date_string)
new_df['dispatch_date_and_time'] = pd.to_datetime(new_df['Dispatch Date and Time'], format=date_string)
new_df['en_route_date_and_time'] = pd.to_datetime(new_df['En route Date and Time'], format=date_string)
new_df['on_scene_date_and_time'] = pd.to_datetime(new_df['On Scene Date and Time'], format=date_string)
new_df['close_date_and_time'] = pd.to_datetime(new_df['Close Date and Time'], format=date_string)

new_df['date'] = pd.to_datetime(new_df['call_date_and_time'])
new_df['hour'] = new_df['date'].dt.hour

# Calculting the differences between the milestones in the process. These are all in min.
# `delta_6_min` will be the column that we will predicit.

new_df['delta_1_min'] = (new_df['entry_date_and_time'] - new_df['call_date_and_time']).dt.total_seconds() / 60
new_df['delta_2_min'] = (new_df['dispatch_date_and_time'] - new_df['entry_date_and_time']).dt.total_seconds() / 60
new_df['delta_3_min'] = (new_df['en_route_date_and_time'] - new_df['dispatch_date_and_time']).dt.total_seconds() / 60
new_df['delta_4_min'] = (new_df['on_scene_date_and_time'] - new_df['en_route_date_and_time']).dt.total_seconds() / 60
new_df['delta_5_min'] = (new_df['close_date_and_time'] - new_df['en_route_date_and_time']).dt.total_seconds() / 60
new_df['delta_6_min'] = (new_df['on_scene_date_and_time'] - new_df['call_date_and_time']).dt.total_seconds() / 60
new_df['delta_7_min'] = (new_df['dispatch_date_and_time'] - new_df['call_date_and_time']).dt.total_seconds() / 60

# Code to fetch the geocoding data for each address. 
# The entire json is stored in the dataframe.
base_url = 'http://api.positionstack.com/v1/forward'
access_key = access_key


new_df['lat_lon_json'] = ''
catch = []
for index, row in new_df.iterrows():
    
    if index % itr == 0:
        print('Fetching Record: {} at {} '.format(index, datetime.now()))
        
    try:
        conn_string = str(base_url + '?access_key=' + str(access_key) + '&query=' + str(row['address_query_string']))
        c = conn_string.replace(' ', '%20')
        data = r.get(c)
        new_df.at[index, 'lat_lon_json'] = data.json()
    except:
        print('Error: {}'.format(index))
        new_df.at[index, 'lat_lon_json'] = -1
        catch.append(index)

#Parses out the json and returns the lat/lon for each address. 
new_df['lat'] = ''
new_df['lon'] = ''
catch = []
for index, row in new_df.iterrows():

    try:
        new_df.at[index, 'lat'] = row['lat_lon_json']['data'][0]['latitude']
        new_df.at[index, 'lon'] = row['lat_lon_json']['data'][0]['longitude']
    except:
        new_df.at[index, 'lat'] = -1
        new_df.at[index, 'lon'] = -1
        print('Error: {}'.format(index))
        catch.append(index)             

new_df = new_df[new_df.lat != -1]

def distance(lat2=None, lon2=None):
    '''
    Haversine Distance: Generates
    the distance between two lat/lon
    points.
    '''
    
    rad_earth = 6373.0
    ref_lat = 36.863140
    ref_lon = -76.015778

    lat1 = radians(ref_lat)
    lon1 = radians(ref_lon)
    lat2_ = radians(lat2)
    lon2_ = radians(lon2)

    dlon = lon2_ - lon1
    dlat = lat2_ - lat1

    part_1 = sin(dlat / 2)**2 + cos(lat1) * cos(lat2_) * sin(dlon / 2)**2
    part_2 = 2 * atan2(sqrt(part_1), sqrt(1 - part_1))

    distance = rad_earth * part_2
    
    return distance

#apply the distance formulat to the lat/lon using the reference point.
new_df2 = new_df.copy()
new_df2['distance_km'] = new_df2.apply(lambda x: distance(x['lat'], x['lon']), axis=1)
new_df2.to_csv('ssie-637-sample-2022-11-26-25k-part-2_random_state_95.csv', index=False)
