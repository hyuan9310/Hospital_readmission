#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:27:50 2018

@author: TIANYING
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

#Data balancing: 输入trainingDF 输出balanced trainingDF 
def resampletrain(df):
    df_majority = df[df.readmitted==0]
    df_minority = df[df.readmitted==1]
    df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=len(df_majority.index),    
                                 random_state=123) 
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled

diabetes = pd.read_csv("diabetic_data.csv", engine="c")
# lower case and underscore for column name
diabetes.columns = diabetes.columns.str.replace("-", "_").str.lower()
diabetes.drop(["weight", "payer_code", "medical_specialty"], axis=1, inplace=True)
diabetes.drop(["encounter_id", "patient_nbr"], axis=1, inplace=True)

# split the columns into continuous and categorical for easier processing
cat_var = ['race','gender','age', 'diag_1','diag_2','diag_3',
           'max_glu_serum','a1cresult','metformin','repaglinide',
           'nateglinide','chlorpropamide','glimepiride','acetohexamide',
           'glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone',
           'acarbose','miglitol','troglitazone','tolazamide','examide',
           'citoglipton','insulin','glyburide_metformin','glipizide_metformin',
           'glimepiride_pioglitazone','metformin_rosiglitazone','metformin_pioglitazone',
           'change','diabetesmed','readmitted']

cont_var = ['admission_type_id','discharge_disposition_id','admission_source_id',
            'time_in_hospital','num_lab_procedures','num_procedures','num_medications',
            'number_outpatient','number_emergency','number_inpatient','number_diagnoses']

# lowercase all elements in dataframe
diabetes[cat_var] = diabetes[cat_var].applymap(lambda x: str(x).lower())
# change to the correct datatype
diabetes[cat_var] = diabetes[cat_var].apply(lambda x: x.astype("category"))
diabetes[cont_var] = diabetes[cont_var].apply(lambda x: x.astype("int"))

# creating binary classification rather than multiclass classification
# any readmission is converted to yes
diabetes.readmitted = diabetes.readmitted.map(lambda x: "yes" if x != "no" else x)
diabetes.readmitted = diabetes.readmitted.map(lambda x: x.lower())

# replaced all ? to actual null
diabetes.replace("?", np.nan, inplace=True)
# creating a dictionary that maintains the order of age
age_key = {item:i+1 for item, i in zip(np.unique(diabetes.age),[i for i in range(len(np.unique(diabetes.age)))])}
# applying that order and convert from categorical to continuous
diabetes.age = diabetes.age.map(lambda x: age_key[x])

plt.figure(figsize=(10,10));

# standardizing the variables with outlier
rs = RobustScaler()
diabetes[cont_var] = rs.fit_transform(diabetes[cont_var])

#sns.countplot(data=diabetes, x='gender', hue='readmitted')
# replace unknown/invalid with null for easier imputation
diabetes.gender.replace("unknown/invalid", np.nan, inplace=True)
# comparing number of records before and after dropping null values
#print(diabetes.shape, diabetes.dropna().shape)

# since we didn't lose too many records, I will be dropping the null values
diabetes.dropna(inplace=True)
# use LabelEncoder to convert all categorical values to numerical values
LE = LabelEncoder()
for var in cat_var:
    diabetes[var] = LE.fit_transform(diabetes[var])

diabetes[cont_var] = diabetes[cont_var].applymap(lambda x: abs(x))
#===================================================
# split the features from the label
features = diabetes.drop("readmitted", axis=1)
label = diabetes.readmitted

# train test split with 80% of the data as the training set
x_train, x_test, y_train, y_test = train_test_split(features, label, train_size = 0.8)
# with all features
logR = LogisticRegression(n_jobs=-1)
logR.fit(x_train, y_train)
logR.score(x_test, y_test)
logR_pred = logR.predict(x_test)

print(classification_report(y_test, logR_pred))

fpr, tpr, thresholds = roc_curve(y_test, logR_pred)
area_u_curve = auc(fpr, tpr)

plt.figure(figsize=(7,7));
plt.plot(fpr, tpr, color='green', label='ROC curve (area = {})'.format(area_u_curve));
plt.plot([0,1], [0,1], linestyle='--');
plt.xlabel("false positive rates");
plt.ylabel("true positive rates");
plt.title('Receiver operating characteristic for logistic regression');
plt.legend(loc='upper left');

# initialize PCA
pca = PCA(n_components=2)
reduced_feature = pca.fit_transform(features)

# initialize kmean with 2 cluster because I have 2 labels
km = KMeans(n_clusters=2, n_jobs=-1)

# fitting PCA-ed features into kmean
km.fit(reduced_feature)

# create a dataframe for the predictions
km_pred = pd.DataFrame(km.predict(reduced_feature))

# mapping colors
km_cmap = {0:'red', 1:'green'}
km_pred['color'] = km_pred[0].replace(km_cmap)

import matplotlib.patches as mpatches

# mapping colors for legend
class_colours = ['red', 'green']
recs = []
for i in range(0,len(class_colours)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
    
# plotting kmean result
classes = ['not readmitted', "readmitted"]
plt.figure(figsize=(10,10));
plt.scatter(reduced_feature[:, 0], 
            reduced_feature[:, 1], 
            c=km_pred.color);
plt.xlabel('principal component 1');
plt.ylabel('principal component 2');
plt.title('k_mean clustering on PCA');
plt.legend(recs, classes);
