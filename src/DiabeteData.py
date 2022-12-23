from sklearn.impute import SimpleImputer
import sklearn
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class DiabeteData:
    
    def __init__(self):
        """
        Our constructor class that instantiates bike rental shop.
        """
    
    def load_data(self):
        df = pd.read_csv("./pima-indians-diabetes.csv")
        self.data_frame = df
        self.y = df['DiabetesPrediction']
        self.X = df.drop(['DiabetesPrediction'], axis=1)
        self.feature_names = ['Pregnancies',
                         'Glucose',
                         'BloodPressure',
                         'SkinThickness',
                         'Insulin',
                         'BMI',
                         'DiabetesPedigreeFunction',
                         'Age']

        return self


    def load_important_data(self):
        df = pd.read_csv("./pima-indians-diabetes.csv")
        self.data_frame = df
        self.y = df['DiabetesPrediction']
        self.X = df.drop(['DiabetesPrediction'], axis=1)
        self.X = self.X.iloc[:,[6]]
        DiabetesPedigreeFunction_col = self.X['DiabetesPedigreeFunction']
        DiabetesPedigreeFunction_col.replace(to_replace=0, value=DiabetesPedigreeFunction_col.mean(), inplace=True)
        self.feature_names = ['DiabetesPedigreeFunction']

        return self

    def data_split(self):

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(self.X, self.y, test_size=0.30,random_state=1)

        self.X_train =X_train
        self.X_test =X_test
        self.y_train =y_train
        self.y_test =y_test

        self.np_X = np.array(self.X)
        self.np_Y = np.array(self.y)
        self.np_X_train = np.array(X_train)
        self.np_y_train = np.array(y_train)
        self.np_X_test = np.array(X_test)
        self.np_y_test = np.array(y_test)

        print("{0:0.2f}% in training set".format((len(X_train)/len(self.data_frame.index)) * 100))
        print("{0:0.2f}% in test set".format((len(X_test)/len(self.data_frame.index)) * 100))
        #print("data has split successfully!")


    def pca(self):
        pca = PCA(n_components=7)
        principalComponents = pca.fit_transform(self.X)
        self.X = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2',
                                                            'principal component 3', 'principal component 4',
                                                            'principal component 5', 'principal component 6',
                                                            'principal component 7'])

    def remove_zero_columns(self):
        self.data_frame.isnull().sum()
        print("# rows in dataframe {0}".format(len(self.data_frame)))
        print("# Zero values in Glucose column: {0}".format(len(self.data_frame.loc[self.data_frame['Glucose'] == 0])))
        print("# Zero values in BloodPressure column: {0}".format(len(self.data_frame.loc[self.data_frame['BloodPressure'] == 0])))
        print("# Zero values in SkinThickness column: {0}".format(len(self.data_frame.loc[self.data_frame['SkinThickness'] == 0])))
        print("# Zero values in Insulin column: {0}".format(len(self.data_frame.loc[self.data_frame['Insulin'] == 0])))
        print("# Zero values in BMI column: {0}".format(len(self.data_frame.loc[self.data_frame['BMI'] == 0])))
        print("# Zero values in DiabetesPedigreeFunction column: {0}".format(len(self.data_frame.loc[self.data_frame['DiabetesPedigreeFunction'] == 0])))
        print("# Zero values in Age column: {0}".format(len(self.data_frame.loc[self.data_frame['Age'] == 0])))
        print("Zero columns have removed successfully!")


    def impute_data(self):
        imputer = SimpleImputer(missing_values=0, strategy='mean')
        self.data_frame.Glucose = imputer.fit_transform(self.data_frame['Glucose'].values.reshape(-1, 1))[:, 0]

        imputer = SimpleImputer(missing_values=0, strategy='mean')
        self.data_frame.BloodPressure = imputer.fit_transform(self.data_frame['BloodPressure'].values.reshape(-1, 1))[:, 0]

        imputer = SimpleImputer(missing_values=0, strategy='mean')
        self.data_frame.SkinThickness = imputer.fit_transform(self.data_frame['SkinThickness'].values.reshape(-1, 1))[:, 0]

        imputer = SimpleImputer(missing_values=0, strategy='mean')
        self.data_frame.Insulin = imputer.fit_transform(self.data_frame['Insulin'].values.reshape(-1, 1))[:, 0]

        imputer = SimpleImputer(missing_values=0, strategy='mean')
        self.data_frame.BMI = imputer.fit_transform(self.data_frame['BMI'].values.reshape(-1, 1))[:, 0]

        #print("data has imputed successfully!")

    def Preprocessing(self):
         # filling zero value with mean
        Glucose_col = self.X['Glucose']
        Glucose_col.replace(to_replace=0, value=Glucose_col.mean(), inplace=True)

        BloodPressure_col = self.X['BloodPressure']
        BloodPressure_col.replace(to_replace=0, value=BloodPressure_col.mean(), inplace=True)

        SkinThickness_col = self.X['SkinThickness']
        SkinThickness_col.replace(to_replace=0, value=SkinThickness_col.mean(), inplace=True)

        Insulin_col = self.X['Insulin']
        Insulin_col.replace(to_replace=0, value=Insulin_col.mean(), inplace=True)

        BMI_col = self.X['BMI']
        BMI_col.replace(to_replace=0, value=BMI_col.mean(), inplace=True)

        DiabetesPedigreeFunction_col = self.X['DiabetesPedigreeFunction']
        DiabetesPedigreeFunction_col.replace(to_replace=0, value=DiabetesPedigreeFunction_col.mean(), inplace=True)

        Age_col = self.X['Age']
        Age_col.replace(to_replace=0, value=Age_col.mean(), inplace=True)

        #print("data has Preprocessed successfully!")



    def Load_data_preprocessing(self):
        # creating outputclasses and features

        self.load_data()

        y = self.data_frame['DiabetesPrediction']
        X = self.data_frame.drop(['DiabetesPrediction'], axis=1)

        # filling zero value with mean
        self.Preprocessing()

        pca = PCA(n_components=7)
        principalComponents = pca.fit_transform(X)
        X = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2',
                                                            'principal component 3', 'principal component 4',
                                                            'principal component 5', 'principal component 6',
                                                            'principal component 7'])

        # Spliting dataset to train and test data %70 train %30 test
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.30,
                                                            random_state=1)
        # MLP
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


