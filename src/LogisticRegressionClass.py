class LogisticRegressionClass:

    def __init__(self):
        """
        Our constructor class that instantiates model.
        """

    def model_fit(self, model, diabete_data):
        model.fit(diabete_data.X_train,
                  diabete_data.y_train)
        self.predict_test = model.predict(diabete_data.X_test)