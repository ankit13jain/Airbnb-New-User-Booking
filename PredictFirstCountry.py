import pandas as pd
import numpy as np
from random import randint
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold

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
    
    #Valid age range between 14 to 90 as per data, otherwise check if its outlier or not
    df['age']=df['age'].apply(lambda x: x if 14<=x<=90 else np.nan)     
    mean = df['age'].mean()
    mean = int(mean)
    df['age']=df['age'].apply(lambda x: mean if np.isnan(x) else x) 
    return df

#This functions generates random value keeping the proportion of possible values in consideration
def weightedRandomHelper(pairs):  
    total = sum(pair[0] for pair in pairs)
    r = randint(1, total)
    for (weight, value) in pairs:
        r -= weight
        if r <= 0: return value

def weightedRandomImputation(df):
    for col in df:
        nan_count=df[col].isnull().sum()
        if col=='age':
            df=handle_outlier_age(df)
            
        # For parameters other then age, impute their missing value using stratified methodology of missing value imputation    
        if nan_count>0 and col!='age': 
            df_counts=df[col].value_counts()
            Total_minus_unknown = 0
            Total_minus_unknown = len(df[col]) - len(df_counts)
            ratio_list=[]
            for i in range(len(df_counts)):
                ratio_list.append(float(df_counts[i])*100/float(Total_minus_unknown))
            min_ratio = min(ratio_list)
            ratio_list = [int(x/min_ratio) for x in ratio_list]
            counts_list=df_counts.index.tolist()
            pairs = list(zip(ratio_list,counts_list))
            df[col]=df[col].apply(lambda x: weightedRandomHelper(pairs) if(pd.isnull(x)) else x)

        # Creating bins for signup_flow parameter
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
    X_train = X_train.drop('timestamp_first_active', 1)
    X_test = X_test.drop('language', 1)
    X_test = X_test.drop('signup_app', 1)
    X_test = X_test.drop('signup_flow', 1)
    X_test = X_test.drop('timestamp_first_active', 1)

    clf = RandomForestClassifier(max_features= 'auto', max_depth = 20, random_state=10, min_samples_split = 4, verbose =1, class_weight = 'balanced', oob_score =False, n_estimators = 100)

    clf.fit(X_train, Y_train)
    print("Importance of the features : ",clf.feature_importances_)
    
    x = [i for i in range(0,len(clf.feature_importances_))]
    plt.xticks(x, list(X_train))
    plt.plot(x, clf.feature_importances_,"ro")
    plt.plot(x, clf.feature_importances_)
    plt.xlabel("Features")
    plt.ylabel("Relevance Factor")
    plt.title("Relevance of the Features as per Random Forest Classifier Model")
    plt.xticks(rotation='vertical')
    plt.show()
    
    Y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, Y_test, sample_weight=None)
    print ("Accuracy using Random Forest Classifier is : %.2f%%" % (accuracy * 100.0))
    print("The confusion matrix is : \n",confusion_matrix(Y_test, Y_pred ))
    print("Mean Absolute error is :",mean_absolute_error(Y_test, Y_pred ))
    print("Evaluation Metrics :\n",classification_report(Y_test, Y_pred ))

def Naivebayes(df,df_test):
    print("\nLearning the Naivebayes Classifier Model...")
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

    gnb = GaussianNB() 
    gnb.fit(X_train, Y_train)
    Y_pred = gnb.predict(X_test)

    predictions = [round(value) for value in Y_pred]

    accuracy = accuracy_score(Y_test, predictions)
    print("Accuracy with NB is : %.2f%%" % (accuracy * 100.0))
    print("The confusion matrix is : \n",confusion_matrix(Y_test, Y_pred ))
    print("Mean Absolute error is :",mean_absolute_error(Y_test, Y_pred ))
    print("Evaluation Metrics :\n",classification_report(Y_test, Y_pred ))

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
    Y_pred = model.predict(X_test)

    predictions = [round(value) for value in Y_pred]

    accuracy = accuracy_score(Y_test, predictions)
    print("Accuracy with XGBoost is : %.2f%%" % (accuracy * 100.0))
    print("The confusion matrix is : \n",confusion_matrix(Y_test, Y_pred ))
    print("Mean Absolute error is :",mean_absolute_error(Y_test, Y_pred ))
    print("Evaluation Metrics :\n",classification_report(Y_test, Y_pred ))
    
def KNNClassifier(df,df_test):
    print("\nLearning the KNN Classifier Model...")
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

    n_neighbors = 300
    #for weights in ['uniform', 'distance']:
    for weights in ['distance']:
    # we create an instance of Neighbours Classifier and fit the data.
        #clf = KNeighborsClassifier(n_neighbors, weights=weights)
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights,algorithm='ball_tree')
        clf.fit(X_train, Y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
        Y_pred = clf.predict(X_test)
        prediction_knn = [round(value) for value in Y_pred]
        accuracy = accuracy_score(Y_test, prediction_knn)
        print("Accuracy with KNN is : %.2f%%" % (accuracy * 100.0))
        
    print("The confusion matrix is : \n",confusion_matrix(Y_test, Y_pred ))
    print("Mean Absolute error is :",mean_absolute_error(Y_test, Y_pred ))
    print("Evaluation Metrics :\n",classification_report(Y_test, Y_pred ))


def ann(df,df_test):

    print("\nLearning the Artificial Neural Network Classifier Model...")
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

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    encoded_Y = encoder.transform(Y_train)
    print(encoded_Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    Y_train = pd.DataFrame(np_utils.to_categorical(encoded_Y))

    encoder = LabelEncoder()
    encoder.fit(Y_test)
    encoded_Y = encoder.transform(Y_test)
    print(encoded_Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    Y_test = pd.DataFrame(np_utils.to_categorical(encoded_Y))
    
    df_encoded = pd.DataFrame(index=range(1,len(X_train)))    
    train = pd.concat([X_train, X_test])

    for col in train:
        if col=='age': 
            bins = [13,20,30,40,50,60,70,80,91]
            group_names = [0,1,2,3,4,5,6,7]
            train['age_bins'] = pd.cut(train['age'], bins, labels=group_names)
            train=train.drop('age',1)
            col = 'age_bins'
        encoder = LabelEncoder()
        encoder.fit(train[col])
        encoded_Col = encoder.transform(train[col])
        df_encoded = pd.concat([df_encoded,pd.DataFrame(np_utils.to_categorical(encoded_Col))],axis=1)
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=len(df_encoded.columns), activation='relu'))
    #model.add(Dense(50, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(df_encoded.values[:len(X_train)], Y_train.values, epochs=40, batch_size=1000)
    scores = model.evaluate(df_encoded.values[:len(X_train)], Y_train.values)
    print("\nTraining Score: %.2f" % (scores[1]*100))
    scores = model.evaluate(df_encoded.values[len(X_train):], Y_test.values)
    print("\nTesting Score: %.2f" % (scores[1]*100))

    Y_pred = model.predict(df_encoded.values[len(X_train):])
    print("The confusion matrix is : \n",confusion_matrix(Y_test.values.argmax(axis=1), Y_pred.argmax(axis=1)))
    print("Mean Absolute error is :",mean_absolute_error(Y_test.values.argmax(axis=1), Y_pred.argmax(axis=1)))
    print("Evaluation Metrics : \n",classification_report(Y_test.values.argmax(axis=1), Y_pred.argmax(axis=1)))

    
###########_____Driver Code______##############

df = pd.read_csv('train_users_2.csv')   #load data

print("Doing Preprocessing")
print("Handling Missing Values")
df = findNA(df)
original_data  = df.copy()
original_data=encodeDate(original_data)   #convert date to the day of the week with Monday=0, Sunday=6
original_data=weightedRandomImputation(original_data) # Missing Value Imputation

df,df_test = train_test_split( df, test_size=0.3, stratify=df['country_destination'])

df=encodeDate(df)   #convert date to the day of the week with Monday=0, Sunday=6
df=weightedRandomImputation(df) # Missing Value Imputation

#preprocess of test
df_test = encodeDate(df_test)
df_test = weightedRandomImputation(df_test)

randomForestDecisionClassifier(df,df_test)
KNNClassifier(df,df_test)
Naivebayes(df,df_test)
ann(df,df_test)
xgboostClassifier(df,df_test)
