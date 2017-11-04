import numpy as np
import pandas as pd

df = pd.read_csv("train_users_2.csv")
print(df.shape)
cnt=0
for col in df:
    for row in df.index:
        if(col=='gender' and df[col][row]=='MALE'):
            df[col][row]='FEMALE'
            cnt=cnt+1
     
print(cnt)
