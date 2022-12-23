import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sklearn.neural_network
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.tree        # For DecisionTreeClassifier class
import sklearn.ensemble    # For RandomForestClassifier class
import sklearn.datasets    # For make_circles
import sklearn.metrics     # For accuracy_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

class  RandomForestClass:

    def __init__(self):
        """
        Our constructor class that instantiates model.
        """

    def model_fit(self, model, diabete_data):
        # mlpc = sklearn.neural_network.MLPClassifier(random_state=0).fit(X_train, y_train)
        model.fit(diabete_data.X_train, diabete_data.y_train)
        # Forecasting on the Unvalidated Model
        self.predict_test  = model.predict(diabete_data.X_test)  # model prediction process over test set

        #accuracy_score(diabete_data.np_Y_test, linear_y_pred)


    def display_training_metrics(self, diabete_data):
        # training metrics
        print("Accuracy:{0:.4f}".format(metrics.accuracy_score(diabete_data.y_test, self.predict_test)))
        print()
        print("Confusion Matrix")
        print(metrics.confusion_matrix(diabete_data.y_test, self.predict_test))
        print()
        print("Classification Report")
        print(metrics.classification_report(diabete_data.y_test, self.predict_test))

        print("Accuracy:{0:.4f}".format(metrics.accuracy_score(diabete_data.y_test, self.predict_test)))
        print()
        print("Confusion Matrix")
        print(metrics.confusion_matrix(diabete_data.y_test, self.predict_test))
        print()
        print("Classification Report")
        print(metrics.classification_report(diabete_data.y_test, self.predict_test))
