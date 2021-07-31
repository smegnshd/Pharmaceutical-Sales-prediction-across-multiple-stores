# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:19:10 2021

@author: Smegn
"""

 #linear algebra and data processing
import numpy as np
import pandas as pd
import datetime

#visualisations
import seaborn as sb
import matplotlib.pyplot as plt
##
import mlflow
import mlflow.sklearn

# data preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

#ml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# math and statistics
from scipy import stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# ignnore warnings
import warnings
warnings.filterwarnings(action="ignore")
#############Load Data################
train_data=pd.read_csv(r'C:\Users\Smegn\Documents\GitHub\Pharmaceutical-Sales-prediction-across-multiple-stores\data\train.csv')


test_data=pd.read_csv(r'C:\Users\Smegn\Documents\GitHub\Pharmaceutical-Sales-prediction-across-multiple-stores\data\test.csv')

store_data=pd.read_csv(r'C:\Users\Smegn\Documents\GitHub\Pharmaceutical-Sales-prediction-across-multiple-stores\data\store.csv')

sample_data=pd.read_csv(r'C:\Users\Smegn\Documents\GitHub\Pharmaceutical-Sales-prediction-across-multiple-stores\data\sample_submission.csv')

print('train set shape:', train_data.shape)
print('test set shape:', test_data.shape)
print('store set shape:', store_data.shape)

###Mising value##

'''6 of the store data columns contain missing values and one column of test data contain missing value.'''

#Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # dtype of missing values
    mis_val_dtype = df.dtypes

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Dtype'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns
missing_values_table(store_data)
missing_values_table(train_data)
missing_values_table(test_data)
###
'''
Before we deside to fill missing value we need to first merge both data sets with the store data
'''
# merge the train/test sets with the stores set
store_train = pd.merge(left = train_data, right = store_data, how = 'inner', left_on = 'Store', right_on = 'Store')
store_test = pd.merge(left = test_data, right = store_data, how = 'inner', left_on = 'Store', right_on = 'Store')
print(store_train.shape)
print(store_test.shape)
######Data Preprocessing###
#null values
store_train.isna().any()
store_test.isna().any()
###missing
missing_values_table(store_train)
missing_values_table(store_test)
####fill mising value
store_test['CompetitionDistance'].fillna(store_test['CompetitionDistance'].median(), inplace =True)
store_test.CompetitionOpenSinceMonth.fillna(0, inplace = True)
store_test.CompetitionOpenSinceYear.fillna(0,inplace=True)
store_test.Promo2SinceWeek.fillna(0, inplace = True)
store_test.Promo2SinceYear.fillna(0,inplace=True)
store_test.PromoInterval.fillna(0,inplace=True)
store_test.Open.fillna(0,inplace=True)
#fill mising value for store_train dataset
store_train['CompetitionDistance'].fillna(store_train['CompetitionDistance'].median(), inplace =True)
store_train.CompetitionOpenSinceMonth.fillna(0, inplace = True)
store_train.CompetitionOpenSinceYear.fillna(0,inplace=True)
store_train.Promo2SinceWeek.fillna(0, inplace = True)
store_train.Promo2SinceYear.fillna(0,inplace=True)
store_train.PromoInterval.fillna(0,inplace=True)
###check missing value###
missing_values_table(store_train)
missing_values_table(store_test)

#####function to plot
def plotvar(df, variable):
    plt.subplot(1,2,2)
    sb.boxplot(df[variable])
    plt.show()
    
#outliers plot
plotvar(store_train,'CompetitionDistance')
plotvar(store_test,'CompetitionOpenSinceMonth')
plotvar(store_train,'CompetitionOpenSinceYear')
plotvar(store_train,'Promo2SinceWeek')
plotvar(store_train,'Promo2SinceYear')
plotvar(store_test,'Open')

####Fix outliers###
def fix_outlier(df, column):
    df[column] = np.where(df[column] > df[column].quantile(0.95), df[column].median(),df[column])
    
    return df[column]
#fix outliers
fix_outlier(store_train, 'CompetitionDistance')
fix_outlier(store_test, 'CompetitionDistance')
fix_outlier(store_train, 'Open')
fix_outlier(store_test, 'Open')
#plot after fixing outliers
plotvar(store_train,'CompetitionDistance')

#####extract new feature from the existing datetime columns  
#change date column to datetime
def feature_generation(data):
    data['Date'] = pd.to_datetime(data.Date)
    data['Month'] = data.Date.dt.month.to_list()
    data['Year'] = data.Date.dt.year.to_list()
    data['Day'] = data.Date.dt.day.to_list()
    data['WeekOfYear'] = data.Date.dt.weekofyear.to_list()
    data['DayOfWeek'] = data.Date.dt.dayofweek.to_list()
    data['weekday'] = 1 # Initialize the column with default value of 1
    data.loc[data['DayOfWeek'] == 5, 'weekday'] = 0
    data.loc[data['DayOfWeek'] == 6, 'weekday'] = 0
    data = data.drop(['Date'], axis = 1)
    return data

store_train_features=feature_generation(store_train)
store_test_features=feature_generation(store_test)
#identify weekends
store_train['is_weekend'] = ((pd.DatetimeIndex(store_train['Date']).dayofweek) // 5 == 1).astype(int)
store_train.head(14)
#Adding salespercustomer column
store_train['SalesperCustomer']=store_train['Sales']/store_train['Customers']
######
######
######
####Defining independent and dependent variables####
#######
store_train_features['SalesperCustomer']=store_train_features['Sales']/store_train_features['Customers']
#X = train_data.drop(['Customers', 'Sales', 'SalesperCustomer'], axis = 1)
store_train_features.corr()
#SInce we want to predict store sales, the target/ dependent variable is sales. For features we
#remove all columns that are strongly correlated to sales. From correlation analysis, we see that
#\"customers\" and \"salespercustomer\" have a strong positive correlation with sale. There we do
#away with these columns.
X = store_train_features.drop(['Customers', 'Sales', 'SalesperCustomer','Open','Promo'], axis = 1)
y=store_train_features['Sales']
#Training and testing split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=15)

#X_train.size, X_test.size, y_train.size, y_test.size
with mlflow.start_run():
####encode###
    le = LabelEncoder()
    le.fit(X_train['StoreType'].astype(str))
    X_train['StoreType']= le.transform(X_train['StoreType'].astype(str))
###
    le = LabelEncoder()
    le.fit(X_train['Assortment'].astype(str))
    X_train['Assortment']= le.transform(X_train['Assortment'].astype(str))

##
    le = LabelEncoder()
    le.fit(X_train['StateHoliday'].astype(str))
    X_train['StateHoliday']= le.transform(X_train['StateHoliday'].astype(str))
##
    le = LabelEncoder()
    le.fit(X_train['PromoInterval'].astype(str))
    X_train['PromoInterval']= le.transform(X_train['PromoInterval'].astype(str))
####
    le = LabelEncoder()
    le.fit(X_test['StoreType'].astype(str))
    X_test['StoreType']= le.transform(X_test['StoreType'].astype(str))
###
    le = LabelEncoder()
    le.fit(X_test['Assortment'].astype(str))
    X_test['Assortment']= le.transform(X_test['Assortment'].astype(str))
##
    le = LabelEncoder()
    le.fit(X_test['StateHoliday'].astype(str))
    X_test['StateHoliday']= le.transform(X_test['StateHoliday'].astype(str))
##
##
    le = LabelEncoder()
    le.fit(X_test['PromoInterval'].astype(str))
    X_test['PromoInterval']= le.transform(X_test['PromoInterval'].astype(str))
###
    regressor = RandomForestRegressor(n_estimators=10, criterion='mse',random_state=0)
    regressor.fit(X_train, y_train)
###predict###
#####################################33
    y_pred = regressor.predict(X_test)
    y_pred_l = pd.DataFrame(y_pred, columns=["sales prediction"])
    print(y_pred_l.head())
    y_test.head()

###
    def rmspe(y, result):
        rmspe = np.sqrt(np.mean( (y - result)**2 ))
        return rmspe
    RFR = RandomForestRegressor(n_estimators=10,  criterion='mse',  max_depth=5,  min_samples_split=2, 
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0,  max_features='auto',max_leaf_nodes=None, 
                                min_impurity_decrease=0.0,  min_impurity_split=None,  bootstrap=True, oob_score=False, n_jobs=4, random_state=31, 
                                verbose=0,  warm_start=False)
#with mlflow.start_run
    params = {'max_depth':(4,6,8,10,12,14,16,20),'n_estimators':(4,8,16,24,48,72,96,128),'min_samples_split':(2,4,6,8,10)}
#scoring_fnc = metrics.make_scorer(rmspe)
#the dimensionality is high, the number of combinations we have to search is enormous, using
#RandomizedSearchCV 
# is a better option then GridSearchCV
    grid = model_selection.RandomizedSearchCV(estimator=RFR,param_distributions=params,cv=5,)
    grid.fit(X_train, y_train)
###Test our RF on the validation set##
#with the optimal parameters i got let's see how it behaves with the validation set\n",
    rfr_val=RandomForestRegressor(n_estimators=128,criterion='mse',  max_depth=20,min_samples_split=10,  min_samples_leaf=1,  min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,bootstrap=True,oob_score=False, n_jobs=4,random_state=None, verbose=0,warm_start=False)
    model_RF_test=rfr_val.fit(X_train,y_train)
    result=model_RF_test.predict(X_test)
    plt.hist(result)
#mean square error
    mean_squared_error(y_test,result)
##Predicting using XGBoost##
