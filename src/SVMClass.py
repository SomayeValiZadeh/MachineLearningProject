import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

class SVMClass:

    def __init__(self):
        """
        Our constructor class that instantiates model.
        """

    def model_fit(self, model, diabete_data):
        model.fit(diabete_data.np_X_train, diabete_data.np_y_train.reshape(-1, ))
        self.predict_test = model.predict(diabete_data.np_X_test)
        #accuracy_score(diabete_data.np_Y_test, linear_y_pred)

    def f_importances(self, model, diabete_data):
        imp = model.coef_
        features_names = ['input1', 'input2']
        imp, names = zip(*sorted(zip(imp, diabete_data.feature_names)))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        plt.show()

    def display_training_metrics(self, diabete_data):
        # training metrics
        print("Accuracy:{0:.4f}".format(metrics.accuracy_score(diabete_data.np_y_test, self.predict_test)))
        print()
        print("Confusion Matrix")
        print(metrics.confusion_matrix(diabete_data.np_y_test, self.predict_test))
        print()
        print("Classification Report")
        print(metrics.classification_report(diabete_data.np_y_test, self.predict_test))
