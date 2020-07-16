import numpy as np
import pandas as pd
import math 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def normCol(col):
    newcolumn = df[col].copy()
    mean = df[col].mean()
    sd = df[col].std()
    newcolumn = (df[col] - mean) / sd
    return newcolumn

def resampletrain(df):
    df_majority = df[df.readmitted==0]
    df_minority = df[df.readmitted==1]
    df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=len(df_majority.index),    
                                 random_state=1) 
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled

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

df['insulin'].replace('No',1,inplace=True)
df['insulin'].replace('Up',2,inplace=True)
df['insulin'].replace('Down',3,inplace=True)
df['insulin'].replace('Steady',4,inplace=True)


df['race'].replace('Hispanic',3,inplace=True)
df['race'].replace('Asian',3,inplace=True)
df['race'].replace('Other',3,inplace=True)
df['race'].replace('Caucasian',1,inplace=True)
df['race'].replace('AfricanAmerican',2,inplace=True)


#diag_1:
diag_1 = df['diag_1']
#Circulatory 390–459, 785
C = list(range(390,460)) + [785]
C = [str(x) for x in C]

for i in C:
    diag_1.replace(i,1,inplace = True)

#Respiratory    460–519, 786
R = list(range(460,520))+[786]
R = [str(i) for i in R ]

for i in R:
    diag_1.replace(i,2, inplace = True)

#Digestive  520–579, 787
D = list(range(520,580))+[787]
D = [str(i) for i in D]
    
for i in D:
    diag_1.replace(i,3,inplace = True)

#Diabetes   250.xx
DB = list(np.arange(250.01,251,0.01))
DB = [round(i,2) for i in DB]
DB = [str(i) for i in DB]
DB += ['250']

for i in DB:
    diag_1.replace(i,4,inplace = True)
    
#Injury 800–999
I = list(range(800,1000))
I = [str(i) for i in I]

for i in I:
    diag_1.replace(i,5,inplace = True)
    
#Musculoskeletal    710–739
M = list(range(710,740))
M = [str(i) for i in M]

for i in M:
    diag_1.replace(i,6,inplace = True)

#Genitourinary  580–629, 788
G = list(range(580,630))+['788']
G = [str(i) for i in G]

for i in G:
    diag_1.replace(i,7,inplace = True)
#Neoplasms 140–239
N = list(range(140,240))
N = [str(i) for i in N]

for i in N:
    diag_1.replace(i, 8, inplace= True)

diagSet = set([1,2,3,4,5,6,7,8])
oindex = df['diag_1'].loc[~df['diag_1'].isin(diagSet)].index.tolist()
for i in oindex:
    df.at[i,'diag_1'] = 9

def combine(feature,rangeList,category):
    index = df[feature].loc[df[feature].isin(rangeList)].index.tolist()
    for i in index:
        df.at[i,feature] = category

combine('num_lab_procedures',list(range(0,11)),10)
combine('num_lab_procedures',list(range(11,21)),20)
combine('num_lab_procedures',list(range(21,31)),30)
combine('num_lab_procedures',list(range(31,41)),40)
combine('num_lab_procedures',list(range(41,51)),50)
combine('num_lab_procedures',list(range(51,61)),60)
combine('num_lab_procedures',list(range(61,71)),70)
combine('num_lab_procedures',list(range(71,81)),80)
combine('num_lab_procedures',list(range(81,91)),90)
combine('num_lab_procedures',list(range(91,150)),100)

combine('num_medications',list(range(0,6)),5)
combine('num_medications',list(range(6,11)),10)
combine('num_medications',list(range(11,16)),15)
combine('num_medications',list(range(16,21)),20)
combine('num_medications',list(range(21,31)),30)
combine('num_medications',list(range(31,41)),40)
combine('num_medications',list(range(41,100)),50)

#reindexing
df = df.reset_index(drop=True)

print('total number of class 0 instances', df['readmitted'].value_counts()[0])
print('total number of class 1 instances', df['readmitted'].value_counts()[1])

label = df['readmitted'].copy()
train = df.drop('readmitted',axis = 1)
X = train
Y = label
rf = RandomForestClassifier()
rf.fit(X, Y)
importance = rf.feature_importances_
result = pd.DataFrame({
    'importance':importance,
    'features':train.columns
})
result = result.sort_values(['importance'] , ascending=0)
print(result)
objects = result['features'].tolist()
y_pos = np.arange(len(objects))
performance = result['importance'].tolist()

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.show();

pos = []
neg = []
acc = []
trainpos = []
trainneg = []
for i in range(1,22):
    select = result['features'][0:i].tolist() + ['readmitted']
    DF = df[select]
    kf = KFold(n_splits=10) 
    kf.get_n_splits(DF)
    accsum = 0
    posum = 0
    negsum = 0
    trainposum = 0
    trainnegsum = 0
    for train_index, test_index in kf.split(DF):
        train, test = DF.iloc[train_index], DF.iloc[test_index]
        x_train = train.drop('readmitted',axis=1)
        y_train = train['readmitted']
        x_test = test.drop('readmitted',axis=1)
        y_test = test['readmitted']
        model = RandomForestClassifier(n_estimators = 1000, n_jobs = -1,class_weight="balanced",min_samples_leaf= 1000)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)
        trainposum += recall_score(y_train, y_train_pred,pos_label=1)
        trainnegsum += recall_score(y_train, y_train_pred,pos_label=0)
        accsum += accuracy_score(y_test, y_pred) 
        posum += recall_score(y_test, y_pred,pos_label=1)
        negsum += recall_score(y_test, y_pred,pos_label=0)
        print(posum)
    accsum /= 10
    posum /= 10
    negsum /= 10
    trainposum /= 10
    trainnegsum /= 10
    acc.append(accsum)
    pos.append(posum)
    neg.append(negsum)
    trainpos.append(trainposum)
    trainneg.append(trainnegsum)


x = list(range(1,22))

plt.plot(x,pos,'-g',label='test recall for TP')
plt.plot(x,trainpos,'-y',label='train recall for TP')
plt.plot(x,neg,'-b',label='test recall for TN')
plt.plot(x,trainneg,label='train recall for TN')
plt.plot(x,acc,'-r',label='testing acc')
plt.legend(loc='lower right')
plt.axis([1,21 , 0.52, 0.625])

plt.xticks(np.arange(min(x), max(x)+1, 1.0))


plt.show();