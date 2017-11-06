import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from random import randint
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
    df['age']=df['age'].apply(lambda x: datetime.now().year-x if x>1900 else x)
    df['age']=df['age'].apply(lambda x: x if 14<=x<=90 else np.nan)
    mode = df['age'].mean()
    mode = int(mode)
    df['age']=df['age'].apply(lambda x: mode if np.isnan(x) else x) 
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
	
        #print("col before ",col,nan_count)
        #print("df col size",len(df[col]))
        if col=='age':
            df=handle_outlier_age(df)
        if nan_count>0 and col!='age':
	    #print col
            df_counts=df[col].value_counts()
            Total_minus_unknown = 0
            Total_minus_unknown = len(df[col]) - len(df_counts)
            ratio_list=[]
            #print(df_counts[0])
            for i in range(len(df_counts)):
                ratio_list.append(float(df_counts[i])*100/float(Total_minus_unknown))
            min_ratio = min(ratio_list)
            ratio_list = [int(x/min_ratio) for x in ratio_list]
            counts_list=df_counts.index.tolist()
            pairs = list(zip(ratio_list,counts_list))
            df[col]=df[col].apply(lambda x: weightedRandomHelper(pairs) if(pd.isnull(x)) else x)
            nan_count=df[col].isnull().sum()
            #print("col after ",col,nan_count)	    	
	    
	if col=='signup_flow':

	    bins = [-1,5,10,15,20,28]
	    group_names = [0,1,2,3,4]
	    df['signup_flow_bins'] = pd.cut(df['signup_flow'], bins, labels=group_names)

    return df


def randomForestDecisionClassifier(df,df_test):
    print("\nLearning the Random Forest Classifier Model...")
    Y_train = df.country_destination
    X_train = df.drop('country_destination', 1)
    X_train = X_train.drop('id', 1)

    #preprocess of test
    Y_test = df_test.country_destination
    X_test = df_test.drop('country_destination', 1)
    X_test = X_test.drop('id', 1)

    # encode Y train
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)

    X_train = X_train.apply(LabelEncoder().fit_transform)
    X_test= X_test.apply(LabelEncoder().fit_transform)

    # Encode Y Test 
    le_t = LabelEncoder()
    Y_test = le_t.fit_transform(Y_test)
    
    #dropping below columns as they do not improve the accuracy based on clf.feature_importances_
    X_train = X_train.drop('language', 1)
    X_train = X_train.drop('signup_app', 1)
    X_train = X_train.drop('signup_flow', 1)
    #X_train = X_train.drop('timestamp_first_active', 1)
    X_test = X_test.drop('language', 1)
    X_test = X_test.drop('signup_app', 1)
    X_test = X_test.drop('signup_flow', 1)
    #X_test = X_test.drop('timestamp_first_active', 1)

    clf = RandomForestClassifier(max_features= 'auto', max_depth = 20, random_state=10, min_samples_split = 4, verbose =1, class_weight = 'balanced', oob_score =False, n_estimators = 100)

    clf.fit(X_train, Y_train)
    print("Importance of the features : ",clf.feature_importances_)
    
    x = [i for i in range(0,len(clf.feature_importances_))]
    plt.xticks(x, list(X_train))
    plt.plot(x, clf.feature_importances_,"ro")
    plt.plot(x, clf.feature_importances_)
    plt.show()

    Y_predict = clf.predict(X_test)
    accuracy = clf.score(X_test, Y_test, sample_weight=None)
    print ("Accuracy using Random Forest Classifier is : %.2f%%" % (accuracy * 100.0))

def xgboostClassifier(df,df_test):
    print("\nLearning the XGBoost Classifier Model...")
    Y_train = df.country_destination
    X_train = df.drop('country_destination', 1)
    X_train = X_train.drop('id', 1)

    #preprocess of test
    Y_test = df_test.country_destination
    X_test = df_test.drop('country_destination', 1)
    X_test = X_test.drop('id', 1)

    # encode Y train
    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)

    X_train = X_train.apply(LabelEncoder().fit_transform)
    X_test= X_test.apply(LabelEncoder().fit_transform)

    # Encode Y Test 
    le_t = LabelEncoder()
    Y_test = le_t.fit_transform(Y_test)

    #dropping columns as they dont improve accuracy
    X_train = X_train.drop('timestamp_first_active', 1)
    X_train = X_train.drop('language', 1)
    X_train = X_train.drop('signup_app', 1)
    X_test = X_test.drop('timestamp_first_active', 1)
    X_test = X_test.drop('language', 1)
    X_test = X_test.drop('signup_app', 1)


    model = XGBClassifier()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(Y_test, predictions)
    print("Accuracy with XGBoost is : %.2f%%" % (accuracy * 100.0))



###########_____Driver Code______##############

df = pd.read_csv('train_users_2.csv')   #load data
print("Doing Preprocessing")
print("Handling Missing Values")
df = findNA(df)


##dfs=np.split(df,[int(shape[0]*0.7)])
##df=dfs[0]   #Training data
##df_test=dfs[1]  #Testing data

df,df_test = train_test_split( df, test_size=0.3, stratify=df['country_destination'])

df=encodeDate(df)   #convert date to the day of the week with Monday=0, Sunday=6
df=weightedRandomImputation(df) # Missing Value Imputation
#df=binning_signup_flow(df)

#preprocess of test
df_test = encodeDate(df_test)
df_test = weightedRandomImputation(df_test)
randomForestDecisionClassifier(df,df_test)
xgboostClassifier(df,df_test)
