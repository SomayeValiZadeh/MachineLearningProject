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


class NeuralNetworkClass:

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


    def cross_validation(self,mlpc, diabete_data):
        mlpc_params = {"alpha": [0.1, 0.01, 0.001],
                       "hidden_layer_sizes": [(32, 24, 12, 10), (32, 24, 12, 8), (30, 22, 10, 6), (100)],
                       "solver": ["lbfgs", "adam", "sgd"],
                       "activation": ["relu", "logistic", "tanh"]
                       }

        mlpc = sklearn.neural_network.MLPClassifier(random_state=0, verbose=False, max_iter=1000,
                                                    batch_size=100)  # ANN model object created

        # Model CV process
        mlpc_cv_model = sklearn.model_selection.GridSearchCV(mlpc, mlpc_params, n_jobs=-1, cv=5,
                                                             verbose=2).fit(diabete_data.X_train, diabete_data.y_train)

        # The best parameter obtained as a result of CV process

        print("The best parameters: " + str(mlpc_cv_model.best_params_))
        # Setting the Final Model with the best parameter
        self.mlpc_tuned = mlpc_cv_model.best_estimator_
