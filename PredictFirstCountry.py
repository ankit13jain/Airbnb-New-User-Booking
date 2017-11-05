import pandas as pd
import numpy as np
from random import randint

def findNA(df):
    df = df.replace(r'\s+', np.nan, regex=True)
    df = df.replace('-unknown-',np.nan, regex=False)
    df = df.replace('Other/Unknown',np.nan, regex=False)
    df = df.dropna(thresh=11) #Ignore the rows with majority Missing Value during Analysis
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
        if nan_count>0 and col!='age':
            df_counts=df[col].value_counts()
            Total_minus_unknown = 0
            for i in range(len(df_counts)):
                Total_minus_unknown +=df_counts[i]
            ratio_list=[]        
            for i in range(len(df_counts)):
                ratio_list.append(float(df_counts[i])*100/float(Total_minus_unknown))
            min_ratio = min(ratio_list)
            ratio_list = [int(x/min_ratio) for x in ratio_list]
            counts_list=df_counts.index.tolist()
            pairs = list(zip(ratio_list,counts_list))
            df[col]=df[col].apply(lambda x: weightedRandomHelper(pairs) if(pd.isnull(x)) else x)
    return df
    

df = pd.read_csv('train_users_2.csv')
df = findNA(df)
shape=df.shape
print(shape)

dfs=np.split(df,[int(shape[0]*0.7)])
df=dfs[0]   #Training data
df_test=dfs[1]  #Testing data
df=weightedRandomImputation(df)
