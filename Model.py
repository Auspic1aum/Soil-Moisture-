#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np 
import pandas as pd 
#pd.set_option("display.max_rows", None, "display.max_columns", None)
from sklearn.model_selection import train_test_split
import pickle
# Train the model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import  r2_score

df1 = pd.read_excel('C:\\Users\\Dell\\Desktop\\Soil Moisture Prediction\\Sm Training feature 2d.xlsx')
#keeping only relevant columns 
df1 = df1[['Date', 'T_max', 'T_min', 'Rainfall_mm', 'Rainfall_NRSC_mm', 'Evp_mm','Soil_Moisture_prev' ,'Soil_Moisture_perc']]
#extract date number 
df1['day_of_month'] = df1['Date'].astype('str').str[-2:]
df1['day_of_month'] = df1['day_of_month'].astype('float')
#extract month num 
df1['month_of_year'] = df1['Date'].astype('str').str[5:7]
df1['month_of_year'] = df1['month_of_year'].astype('float')
#keeping only relevant columns 
df2 = df1[[ 'day_of_month', 'month_of_year','T_max', 'T_min', 'Rainfall_mm', 'Rainfall_NRSC_mm', 'Evp_mm','Soil_Moisture_prev', 'Soil_Moisture_perc' ]]
#dropping Nulls 
df2.dropna(inplace=True)
#remove outliers from all such rows which have atleast one outlier 
from scipy import stats

df3 = df2[(np.abs(stats.zscore(df2)) < 3).all(axis=1)]
# Separate features and labels
# After separating the dataset, we now have numpy arrays named **X** containing the features, and **y** containing the labels.
X, y = df3[['day_of_month', 'month_of_year','T_max', 'T_min', 'Rainfall_mm', 'Rainfall_NRSC_mm','Evp_mm','Soil_Moisture_prev']].values, df3['Soil_Moisture_perc'].values

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
hypem_list = [6,7,8,9,10,11]
r2_list = {}
for i in hypem_list:
    pipeline_rf = Pipeline(steps=[('scaler', StandardScaler()),('regressor', RandomForestRegressor(n_estimators = i))])
    model_rf = pipeline_rf.fit(X_train, y_train)
    predictions_rf = model_rf.predict(X_test)
    r2 = r2_score(y_test, predictions_rf)
    r2_list[i] = r2
    #maximum r2_score degree 
    x = max(zip(r2_list.values(), r2_list.keys()))[1]
    
    
# Create preprocessing and training pipeline
pipeline_rf = Pipeline(steps=[('scaler', StandardScaler()),('regressor', RandomForestRegressor(n_estimators = x))])

# fit the pipeline to train a linear regression model on the training set
model_rf = pipeline_rf.fit(X_train, y_train) 

#Saving Model to disc 
pickle.dump(model_rf, open('model.pkl', 'wb'))

# In[ ]:




