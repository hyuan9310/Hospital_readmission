import numpy as np
import pandas as pd
import math 
import heapq

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def normCol(col):
    newcolumn = df[col].copy()
    mean = df[col].mean()
    sd = df[col].std()
    newcolumn = (df[col] - mean) / sd
    return newcolumn

df = pd.read_csv("diabetic_data.csv")
df.drop(['encounter_id','payer_code','weight','diag_2','diag_3','medical_specialty'], axis=1, inplace=True)
for i in df.columns:
	if df[i].value_counts().max() > 91000:
		df.drop([i], axis=1, inplace=True)
a = df.index[df['race'] == '?'].tolist()
b = df.index[df['gender'] == 'Unknown/Invalid'].tolist()
removeRow = a + b
df.drop(removeRow,axis=0, inplace=True)
#remove duplicated patients record
l = set()
removeL = []
for index, row in df.iterrows():
    if row['patient_nbr'] not in l:
        l.add(row['patient_nbr'])
    else:
        removeL.append(index)
df = df.drop(removeL,axis=0)
df = df.drop('patient_nbr',axis=1)

df['readmitted'] = np.where(df['readmitted'] == '<30',1,0)
df['diabetesMed'] = np.where(df['diabetesMed'] == 'Yes',1,0)
df['change'] = np.where(df['change'] == 'Ch',1,0)
df['glipizide'] = np.where(df['glipizide'] == 'No',0,1)
df['metformin'] = np.where(df['metformin'] == 'No',0,1)
df['A1Cresult'] = np.where(df['A1Cresult'] == 'None',0,1)
df['gender'] = np.where(df['gender'] == 'Male',1,0)
#df['race'] = np.where(df['race'] == 'Caucasian',1,0)

df['age'].replace('[0-10)',5,inplace=True)
df['age'].replace('[10-20)',15,inplace=True)
df['age'].replace('[20-30)',25,inplace=True)
df['age'].replace('[30-40)',35,inplace=True)
df['age'].replace('[40-50)',45,inplace=True)
df['age'].replace('[50-60)',55,inplace=True)
df['age'].replace('[60-70)',65,inplace=True)
df['age'].replace('[70-80)',75,inplace=True)
df['age'].replace('[80-90)',85,inplace=True)
df['age'].replace('[90-100)',95,inplace=True)

df['admission_type_id'].replace(1,'ad_type_emergency',inplace=True)
df['admission_type_id'].replace(2,'ad_type_urgent',inplace=True)
df['admission_type_id'].replace(3,'ad_type_elective',inplace=True)
df['admission_type_id'].replace(4,'ad_type_other',inplace=True)
df['admission_type_id'].replace(5,'ad_type_other',inplace=True)
df['admission_type_id'].replace(6,'ad_type_other',inplace=True)
df['admission_type_id'].replace(7,'ad_type_other',inplace=True)
df['admission_type_id'].replace(8,'ad_type_other',inplace=True)
one_hot = pd.get_dummies(df['admission_type_id'])
df = df.join(one_hot)
df = df.drop('admission_type_id',axis = 1)

l_ddi = list(range(7,30))
l_ddi += [2,4,5]
for i in l_ddi:
	df['discharge_disposition_id'].replace(i,'d_d_other',inplace=True)
df['discharge_disposition_id'].replace(1,'d_d_home',inplace=True)
df['discharge_disposition_id'].replace(3,'d_d_SNF',inplace=True)
df['discharge_disposition_id'].replace(6,'d_d_homehealthservice',inplace=True)
one_hot_2 = pd.get_dummies(df['discharge_disposition_id'])
df = df.join(one_hot_2)
df = df.drop('discharge_disposition_id',axis = 1)

l_asi = list(range(2,7)) + list(range(8,27))
for i in l_asi:
	df['admission_source_id'].replace(i,'a_s_other',inplace=True)
df['admission_source_id'].replace(1,'a_s_Physician_ref',inplace=True)
df['admission_source_id'].replace(7,'a_s_emergency_room',inplace=True)
one_hot_3 = pd.get_dummies(df['admission_source_id'])
df = df.join(one_hot_3)
df = df.drop('admission_source_id',axis = 1)


df['insulin'].replace('No','insulin_No',inplace=True)
df['insulin'].replace('Up','insulin_Up',inplace=True)
df['insulin'].replace('Down','insulin_Down',inplace=True)
df['insulin'].replace('Steady','insulin_Steady',inplace=True)
one_hot_1 = pd.get_dummies(df['insulin'])
df = df.join(one_hot_1)
df = df.drop('insulin',axis = 1)

df['race'].replace('Hispanic','race_other',inplace=True)
df['race'].replace('Asian','race_other',inplace=True)
df['race'].replace('Other','race_other',inplace=True)
one_hot_4 = pd.get_dummies(df['race'])
df = df.join(one_hot_4)
df = df.drop('race',axis = 1)

#diag_1:
diag_1 = df['diag_1']
#Circulatory 390–459, 785
C = list(range(390,460)) + [785]
C = [str(x) for x in C]

for i in C:
    diag_1.replace(i,'Circulatory',inplace = True)

#Respiratory	460–519, 786
R = list(range(460,520))+[786]
R = [str(i) for i in R ]

for i in R:
    diag_1.replace(i,'Respiratory', inplace = True)

#Digestive	520–579, 787
D = list(range(520,580))+[787]
D = [str(i) for i in D]
    
for i in D:
    diag_1.replace(i,'Digestive',inplace = True)

#Diabetes	250.xx
DB = list(np.arange(250.01,251,0.01))
DB = [round(i,2) for i in DB]
DB = [str(i) for i in DB]
DB += ['250']

for i in DB:
    diag_1.replace(i,'Diabetes',inplace = True)
    
#Injury	800–999
I = list(range(800,1000))
I = [str(i) for i in I]

for i in I:
    diag_1.replace(i,'Injury',inplace = True)
    
#Musculoskeletal	710–739
M = list(range(710,740))
M = [str(i) for i in M]

for i in M:
    diag_1.replace(i,'Musculoskeletal',inplace = True)

#Genitourinary	580–629, 788
G = list(range(580,630))+['788']
G = [str(i) for i in G]

for i in G:
    diag_1.replace(i,'Genitourinary',inplace = True)
#Neoplasms 140–239
N = list(range(140,240))
N = [str(i) for i in N]

for i in N:
    diag_1.replace(i, 'Neoplasms', inplace= True)

diagSet = set(['Circulatory','Respiratory','Digestive','Diabetes','Injury','Musculoskeletal','Genitourinary','Neoplasms'])
oindex = df['diag_1'].loc[~df['diag_1'].isin(diagSet)].index.tolist()
for i in oindex:
    df.at[i,'diag_1'] = 'other_diag'

one_hot_5 = pd.get_dummies(df['diag_1'])
df = df.join(one_hot_5)
df = df.drop('diag_1',axis = 1)



df['age'] = normCol('age')
df['time_in_hospital'] = normCol('time_in_hospital')
df['num_lab_procedures'] = normCol('num_lab_procedures')
df['num_procedures'] = normCol('num_procedures')
df['num_medications'] = normCol('num_medications')
df['number_outpatient'] = normCol('number_outpatient')
df['number_emergency'] = normCol('number_emergency')
df['number_inpatient'] = normCol('number_inpatient')
df['number_diagnoses'] = normCol('number_diagnoses')

#reindexing
df = df.reset_index(drop=True)
print('total number of class 0 instances', df['readmitted'].value_counts()[0])
print('total number of class 1 instances', df['readmitted'].value_counts()[1])
#******************************__KNN__*****************************  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
from sklearn.utils import resample
from sklearn.metrics import classification_report

def PCC(x,y): 
    sum_sq_x = 0
    sum_sq_y = 0 
    sum_coproduct = 0
    mean_x = 0
    mean_y = 0
    
    N = len(x)
    
    for i in range(N):
        
        sum_sq_x += x[i] * x[i]
        sum_sq_y += y[i] * y[i]
        sum_coproduct += x[i] * y[i]
        mean_x += x[i]
        mean_y += y[i]
        
    mean_x = mean_x / N
    mean_y = mean_y / N
    pop_sd_x = np.sqrt((sum_sq_x/N) - (mean_x * mean_x))
    pop_sd_y = np.sqrt((sum_sq_y / N) - (mean_y * mean_y))
    cov_x_y = (sum_coproduct / N) - (mean_x * mean_y)
    correlation = cov_x_y / (pop_sd_x * pop_sd_y)
    
    return correlation

##imbalanced data copy 1/0
def resampletrain(df):
    df_majority = df[df.readmitted==0]
    df_minority = df[df.readmitted==1]
    df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=len(df_majority.index),    
                                 random_state=1) 
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled

#PCC
label = df['readmitted'].copy()
train = df.drop('readmitted',axis = 1)
abspcc = []
features = []
for i in train.columns:
    r = PCC(df[i].values,label.values)
    abspcc.append(abs(r))
    features.append(i)
    result = pd.DataFrame({'features':features, '|r|':abspcc})
    result = result.sort_values(['|r|'] , ascending=0)
print(result)

pos = []
neg = []
acc = []
trainpos = []
trainneg = []

#test KNN parameter-neighbors
for n_neighbors in range(1,50,2):#get first related 20 features
#    select = result['features'][0:i].tolist() + ['readmitted']
    DF = df
#    
#    kkk = featuresPcc[:i]
#    select = x.iloc[:,kkk].columns.values.tolist() + ['readmitted']
#    DF = df[select]
    kfolds = 5
    kf = KFold(n_splits=kfolds) 
    kf.get_n_splits(DF)
    accsum = 0
    posum = 0
    negsum = 0
    trainposum = 0
    trainnegsum = 0
    for train_index, test_index in kf.split(DF):
        train, test = DF.iloc[train_index], DF.iloc[test_index]
        #imbalanced data
        train = resampletrain(train)
        x_test = test.drop('readmitted',axis=1)
        y_test = test['readmitted']
        x_train = train.drop('readmitted',axis=1)
        y_train = train['readmitted']
        knn =KNeighborsClassifier(n_neighbors = 25)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        y_train_pred = knn.predict(x_train)
        trainposum += recall_score(y_train, y_train_pred,pos_label=1)
        trainnegsum += recall_score(y_train, y_train_pred,pos_label=0)
        accsum += accuracy_score(y_test, y_pred) 
        posum += recall_score(y_test, y_pred,pos_label=1)
        negsum += recall_score(y_test, y_pred,pos_label=0)

    accsum /= kfolds
    posum /= kfolds
    negsum /= kfolds
    trainposum /= kfolds
    trainnegsum /=kfolds
#    acc.append(accsum)
#    pos.append(posum)
#    neg.append(negsum)
#    trainpos.append(trainposum)
#    trainneg.append(trainnegsum)
    KNeighbors = pd.DataFrame({'n_neighbors':n_neighbors,'label_1_recall':posum})

KNeighbors = KNeighbors.sort_values(['label_1_recall'],ascending=0)
print(KNeighbors)

#features selection:
for i in range(1,20):#get first related 20 features
    select = result['features'][0:i].tolist() + ['readmitted']
    DF = df[select]
    kfolds = 10
    kf = KFold(n_splits=kfolds) 
    kf.get_n_splits(DF)
    accsum = 0
    posum = 0
    negsum = 0
    trainposum = 0
    trainnegsum = 0
    for train_index, test_index in kf.split(DF):
        train, test = DF.iloc[train_index], DF.iloc[test_index]
        #imbalanced data
        train = resampletrain(train)
        x_test = test.drop('readmitted',axis=1)
        y_test = test['readmitted']
        x_train = train.drop('readmitted',axis=1)
        y_train = train['readmitted']
        knn =KNeighborsClassifier(n_neighbors = 53)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        y_train_pred = knn.predict(x_train)
        trainposum += recall_score(y_train, y_train_pred,pos_label=1)
        trainnegsum += recall_score(y_train, y_train_pred,pos_label=0)
        accsum += accuracy_score(y_test, y_pred) 
        posum += recall_score(y_test, y_pred,pos_label=1)
        negsum += recall_score(y_test, y_pred,pos_label=0)

    accsum /= kfolds
    posum /= kfolds
    negsum /= kfolds
    trainposum /= kfolds
    trainnegsum /=kfolds
    acc.append(accsum)
    pos.append(posum)
    neg.append(negsum)
    trainpos.append(trainposum)
    trainneg.append(trainnegsum)

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))

import matplotlib.pyplot as plt

x = list(range(1,20))
plt.plot(x,pos,'-g',label='test recall for TP')
plt.plot(x,trainpos,'-y',label='train recall for TP')
plt.plot(x,neg,'-b',label='test recall for TN')
plt.plot(x,trainneg,label='train recall for TN')
plt.plot(x,acc,'-r',label='testing acc')
plt.legend(loc='lower right')
plt.axis([1,20, 0, 1])

plt.xticks(np.arange(min(x), max(x)+1, 1.0))

plt.show();

#Test KNN accuracy and recall score
#top 12 features and k neighbors equal to 53
select = result['features'][0:12].tolist() + ['readmitted']
DF = df[select]
kfolds = 10
kf = KFold(n_splits=kfolds) 
kf.get_n_splits(DF)
accsum = 0
posum = 0
negsum = 0
trainposum = 0
trainnegsum = 0
for train_index, test_index in kf.split(DF):
    train, test = DF.iloc[train_index], DF.iloc[test_index]
    #imbalanced data
    train = resampletrain(train)
    x_test = test.drop('readmitted',axis=1)
    y_test = test['readmitted']
    x_train = train.drop('readmitted',axis=1)
    y_train = train['readmitted']
    knn =KNeighborsClassifier(n_neighbors = 53)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    y_train_pred = knn.predict(x_train)
    trainposum += recall_score(y_train, y_train_pred,pos_label=1)
    trainnegsum += recall_score(y_train, y_train_pred,pos_label=0)
    accsum += accuracy_score(y_test, y_pred) 
    posum += recall_score(y_test, y_pred,pos_label=1)
    negsum += recall_score(y_test, y_pred,pos_label=0)

accsum /= kfolds
posum /= kfolds
negsum /= kfolds
trainposum /= kfolds
trainnegsum /=kfolds
acc.append(accsum)
pos.append(posum)
neg.append(negsum)
trainpos.append(trainposum)
trainneg.append(trainnegsum)

print('recall_socre_label_1:',pos[11])
print('recall_socre_label_0:',pos[11])
print('feature_selections:\n',result.iloc[:12,0])
