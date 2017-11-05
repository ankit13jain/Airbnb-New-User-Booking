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

            print(ratio_list[0])
            print(ratio_list[1])
            print(ratio_list[2])
            counts_list=df_counts.index.tolist()
            pairs = list(zip(ratio_list,counts_list))
            df[col]=df[col].apply(lambda x: weightedRandomHelper(pairs) if(pd.isnull(x)) else x)
            nan_count=df[col].isnull().sum()
            print("after ",nan_count)
                
                
    return df
    

df = pd.read_csv('train_users_2.csv')
df = findNA(df)
shape=df.shape
print(shape)

dfs=np.split(df,[int(shape[0]*0.7)])
df=dfs[0]   #Training data
df_test=dfs[1]  #Testing data
df=weightedRandomImputation(df)


####delete below
##nan_count=df['gender'].isnull().sum()
##female=df['gender'].value_counts()['FEMALE']
##male=df['gender'].value_counts()['MALE']
##other=df['gender'].value_counts()['OTHER']
##
##
##
##print("other : ", other, " MALE : ", male, " FEMALE : ", female, " Unknown : ",nan_count)
##print("TOTAL : ",other+male+female+nan_count)
##
##Total = other+male+female+nan_count
##Total_minus_unknown = other+male+female
##
##male_ratio = float(male)*100/float(Total_minus_unknown)
##female_ratio = float(female)*100/float(Total_minus_unknown)
##other_ratio = float(other)*100/float(Total_minus_unknown)
##kfactor = min(other_ratio,male_ratio,female_ratio)
##print("Other Ratio : ", other_ratio/kfactor, " MALE Ratio : ", male_ratio/kfactor, " FEMALE : ", female_ratio/kfactor)
##
##    ###delete above

for col in df:
        nan_count=df[col].isnull().sum()
        print("col ",nan_count)
