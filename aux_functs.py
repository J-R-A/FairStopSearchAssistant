import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.base import TransformerMixin, BaseEstimator 


# Function to clean data before modeling

def FeatFilterDrop(df):
    
    new_df = df.copy()
    new_df.drop(['Subject Race Code', 
             'Subject Ethnicity Code',
             'Subject Sex Code',
             'Vehicle Searched Indicator',
             'Reporting Officer Identification ID'],
             axis = 1, inplace = True)
    
    numbers_filter = ~new_df['Subject Age'].astype('str').str.contains('^[1-9]\d*(\.\d+)?$') # Choosing just numeric values
    new_df.loc[numbers_filter,'Subject Age'] = np.NaN
    new_df['Subject Age'] = new_df['Subject Age'].astype('float')
    max_age_filter = new_df['Subject Age'] > 100
    new_df.loc[max_age_filter,'Subject Age'] = np.NaN
      
    return new_df


def CreateLocDept(df):
    new_df = df.copy()
    new_df['Intervention Location Name'] = new_df['Intervention Location Name'].str.lower().str.replace(' ','')
    new_df['Department Name'] = new_df['Department Name'].str.lower().str.replace(' ','')
    names_filter = new_df['Department Name'] != new_df['Intervention Location Name']
    new_df['LocDept'] = new_df['Department Name'] 
    new_df.loc[names_filter,'LocDept'] = new_df['Department Name'] + new_df['Intervention Location Name']
    new_df = new_df.drop(['Intervention Location Name','Department Name'],axis = 1)
    
    return new_df



def GetHours(df):
    new_df = df.copy()
    new_df['Intervention DateTime'] = pd.to_datetime(new_df['Intervention DateTime'],utc=True).dt.tz_localize(None).copy()
    hours = new_df['Intervention DateTime'].dt.hour
    new_df['s_hours'] = np.sin(2 * np.pi * hours / hours.max())
    new_df['c_hours'] = np.cos(2 * np.pi * hours / hours.max())
    new_df = new_df.drop('Intervention DateTime',axis = 1)
    
    return new_df



def CatAges(df):
    bSize = 5
    new_df = df.copy()
    new_df['Subject Age'] = pd.cut(new_df['Subject Age'], np.arange(0,105,bSize)).astype('str')
    #new_X_train['Age'] = np.sin(2 * np.pi * new_X_train.Age / new_X_train.Age.max())
    
    return new_df


def FeatCap(df):
    new_df = df.copy()
    feats = ['Intervention Reason Code','Statute Code Description','Search Authorization Code']
    new_df[feats] = new_df[feats].apply(lambda x: x.astype(str).str.lower().str.replace(' ',''))
    
    return new_df



# Custom Transformers

class LocationTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.Loc = None
        
    def fit(self,X,Y):
        self.Loc = self._get_Loc(X, Y)
        return self
    
    def transform(self, X):
        best_loc = self.Loc
        new_X = X.copy()
        new_X.loc[~new_X.LocDept.isin(best_loc),'LocDept'] = 'other'
        
        return new_X
    
    def _get_Loc(self,X,Y): 
        nBest = 75
        new_X = X.copy()
        new_Y = Y.copy()
        loc_Contraband = pd.DataFrame()
        new_X['Contraband Indicator'] = new_Y['Contraband Indicator']
        loc_Contraband['TotalC'] = new_X.groupby('LocDept')['Contraband Indicator'].sum()
        loc_Contraband['Prop'] = new_X.groupby('LocDept')['Contraband Indicator'].sum() / new_X.groupby('LocDept')['Contraband Indicator'].count()
        loc_Contraband = loc_Contraband.loc[loc_Contraband['TotalC'] > 50,:]
        best_loc = loc_Contraband.Prop.sort_values(ascending=False)[:nBest].index
        
        return best_loc
    
    
class NanImputer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.ModeValues = None
        
    def fit(self,X,y=0):
        self.ModeValues = self.get_mode(X)
        return self
    
    def transform(self,X):
        modes_Df = self.ModeValues
        new_X = X.copy()
        for col in new_X:
            if pd.isnull(new_X[col]).sum() != 0:
                value = modes_Df[col].values[0]
                new_X.loc[pd.isnull(new_X[col]),col] = value
        return new_X
    
    def get_mode(self,X): 
        new_X = X.copy()
        return new_X.mode()



    
# Resampling functions

# Resampling functions
def up_sample(X_train,y_train):
    # Separate majority and minority classes
    X_majority = X_train[y_train['Contraband Indicator']==False]
    X_minority = X_train[y_train['Contraband Indicator']==True]
    
    Y_majority = y_train[y_train['Contraband Indicator']==False]
    Y_minority = y_train[y_train['Contraband Indicator']==True]
    
    # Upsample minority classes
    X_minority_upsampled = resample(X_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=X_majority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
    
    Y_minority_upsampled = resample(Y_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=Y_majority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
    
    # Combine majority class with upsampled minority class
    new_X_train = pd.concat([X_majority, X_minority_upsampled]).sort_index()
    new_Y_train = pd.concat([Y_majority, Y_minority_upsampled]).sort_index()
    # Display new class counts
    return  new_X_train , new_Y_train



def down_sample(X_train,y_train):
    # Separate majority and minority classes
    X_majority = X_train[y_train['Contraband Indicator']==False]
    X_minority = X_train[y_train['Contraband Indicator']==True]
    
    Y_majority = y_train[y_train['Contraband Indicator']==False]
    Y_minority = y_train[y_train['Contraband Indicator']==True]
    
    # Upsample minority classes
    X_majority_downsampled = resample(X_majority, 
                                 replace=False,     # sample with replacement
                                 n_samples=X_minority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
    
    Y_majority_downsampled = resample(Y_majority, 
                                 replace=False,     # sample with replacement
                                 n_samples=Y_minority.shape[0],    # to match majority class
                                 random_state=123) # reproducible results
    
    # Combine majority class with upsampled minority class
    new_X_train = pd.concat([X_minority, X_majority_downsampled]).sort_index()
    new_Y_train = pd.concat([Y_minority, Y_majority_downsampled]).sort_index()
    # Display new class counts
    return  new_X_train , new_Y_train

