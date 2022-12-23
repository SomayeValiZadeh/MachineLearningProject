import datetime
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import sklearn
from sklearn.model_selection import train_test_split
from src.DiabeteData import DiabeteData

class utility:

    def display_feature_importance( model, diabete_data):
        """
        Displays the bikes currently available for rent in the shop.
        """

        feature_col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        predicted_class_names = ['DiabetesPrediction']

        X = diabete_data.data_frame[feature_col_names].values # these are factors for the prediction
        y = diabete_data.data_frame[predicted_class_names].values # this is what we want to predict

        diabete_data.data_frame['DiabetesPrediction'] = y


        score = model.score(diabete_data.data_frame[feature_col_names].values, diabete_data.data_frame['DiabetesPrediction'].values)

        w0 = model.intercept_[0]
        w = model.coef_[0]

        feature_importance = pd.DataFrame(feature_col_names, columns = ["feature"])
        feature_importance["importance"] = pow(math.e, w)
        feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)
        

        ax = feature_importance.plot.barh(x='feature', y='importance')
        plt.show()



    def display_training_metrics(lr_class, diabete_data):
        # training metrics
        print("Accuracy:{0:.4f}".format(metrics.accuracy_score(diabete_data.y_test, lr_class.predict_test)))
        print()
        print("Confusion Matrix")
        print(metrics.confusion_matrix(diabete_data.y_test, lr_class.predict_test))
        print()
        print("Classification Report")
        print(metrics.classification_report(diabete_data.y_test, lr_class.predict_test))


        




