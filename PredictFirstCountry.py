import pandas as pd
import numpy as np
from random import randint
from datetime import datetime

def findNA(df):
    df = df.replace(r'\s+', np.nan, regex=True)
    df = df.replace('-unknown-',np.nan, regex=False)
    df = df.replace('Other/Unknown',np.nan, regex=False)
    df = df.dropna(thresh=11) #Ignore the rows with majority Missing Value during Analysis
    return df

def encodeDate(df):
    df['date_account_created']=pd.to_datetime(df['date_account_created']).dt.dayofweek
    df['date_first_booking']=pd.to_datetime(df['date_first_booking']).dt.dayofweek
    return df

def handle_outlier_age(df):
    df['age']=df['age'].apply(lambda x: x-datetime.now().year if x>1900 else x)
    df['age']=df['age'].apply(lambda x: x if 14<=x<=90 else np.nan)
    return df

def weightedRandomHelper(pairs):
    total = sum(pair[0] for pair in pairs)
    r = randint(1, total)
    for (weight, value) in pairs:
        r -= weight
        if r <= 0: return value

def weightedRandomImputation(df):
    for col in df:
        nan_count=df[col].isnull().sum()
        print("col before ",col,nan_count)
        print("df col size",len(df[col]))
        if col=='age':
            df=handle_outlier_age(df)
        if nan_count>0 and col=='age':
            df_counts=df[col].value_counts()
            Total_minus_unknown = 0
            Total_minus_unknown = len(df[col]) - len(df_counts)
            ratio_list=[]
            print(df_counts[0])
            for i in range(len(df_counts)):
                ratio_list.append(float(df_counts[i])*100/float(Total_minus_unknown))
            min_ratio = min(ratio_list)
            ratio_list = [int(x/min_ratio) for x in ratio_list]
            counts_list=df_counts.index.tolist()
            pairs = list(zip(ratio_list,counts_list))
            df[col]=df[col].apply(lambda x: weightedRandomHelper(pairs) if(pd.isnull(x)) else x)
            nan_count=df[col].isnull().sum()
            print("col after ",col,nan_count)
    return df
    

df = pd.read_csv('train_users_2.csv')   #load data
df = findNA(df)
shape=df.shape
print(shape)

dfs=np.split(df,[int(shape[0]*0.7)])
df=dfs[0]   #Training data
df_test=dfs[1]  #Testing data
df=encodeDate(df)   #convert date to the day of the week with Monday=0, Sunday=6
df=weightedRandomImputation(df)
