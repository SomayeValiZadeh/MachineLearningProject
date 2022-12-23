import sklearn
from sklearn.linear_model import LogisticRegression

from src.DiabeteData import DiabeteData
from src.Utility import utility
from src.LogisticRegressionClass import LogisticRegressionClass
from src.SVMClass import SVMClass
from src.NeuralNetworkClass import NeuralNetworkClass
from src.DecisionTreeClass import DecisionTreeClass
from src.RandomForestClass import RandomForestClass

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import sklearn.neural_network
import numpy as np



def main():
    diabete_data = DiabeteData()
    lr_class = LogisticRegressionClass()
    svm_class = SVMClass()
    neural_network_class = NeuralNetworkClass()
    decision_tree_class = DecisionTreeClass()
    random_forest_class = RandomForestClass()
    while True:
        print("""
        ====== Machine Learning Project =======
        1. Read Data And PreProcessing
        2. Request for Logist Regression
        3. Request for Support Vector Machine(SVM)
        4. Request for Neural Network (NN)
        5. Request for Neural Network (Best model)
        6. Request for Decision Tree 
        7. Request for Random Forest 
        8. Feature importance for Logist Regression
        9. Exit
        """)

        choice = input("Enter choice: ")
        try:
            choice = int(choice)
        except ValueError:
            print("That's not an int!")
            continue

        if choice == 1:
            print("******* Data Load and Preprocessing *******")
            diabete_data.load_data()
            diabete_data.Preprocessing()
            diabete_data.data_split()

        elif choice == 2:

            diabete_data.load_data()
            diabete_data.Preprocessing()
            diabete_data.data_split()
            lr_model = LogisticRegression(C=1, random_state=0, max_iter=1000)
            lr_class.model_fit(lr_model, diabete_data)
            print("******* LogisticRegression *******")
            utility.display_training_metrics(lr_class, diabete_data)
            #utility.display_feature_importance(lr_model, diabete_data)

        elif choice == 3:

            diabete_data.load_data()
            diabete_data.Preprocessing()
            diabete_data.data_split()
            svm_model = sklearn.svm.SVC(random_state=0, kernel='linear')
            svm_class.model_fit(svm_model, diabete_data)
            print("******* Support Vector Machine *******")
            svm_class.display_training_metrics(diabete_data)

        elif choice == 4:

            diabete_data.Load_data_preprocessing()
            mlpc_primitive = sklearn.neural_network.MLPClassifier(random_state=0, max_iter=1000)#.fit(diabete_data.X_train, diabete_data.y_train)
            neural_network_class.model_fit(mlpc_primitive, diabete_data)
            print("******* Neural Network. *******")
            neural_network_class.display_training_metrics( diabete_data)

        elif choice == 5:
            print("******* Neural Network(PCA and cross validation) *******")
            diabete_data.Load_data_preprocessing()
            mlpc = sklearn.neural_network.MLPClassifier(random_state = 0, verbose=False, max_iter = 1000 ,batch_size = 100) # ANN model object created
            neural_network_class.cross_validation(mlpc, diabete_data)
            neural_network_class.model_fit(neural_network_class.mlpc_tuned, diabete_data)
            print("******* Neural Network. *******")
            neural_network_class.display_training_metrics( diabete_data)

        elif choice == 6:
            print("******* DecisionTree *******")
            diabete_data.load_data()
            diabete_data.Preprocessing()
            diabete_data.data_split()
            classifier = DecisionTreeClassifier(random_state = 0, criterion='gini', splitter='best', max_depth=5)
            decision_tree_class.model_fit(classifier, diabete_data)
            print("******* Neural Network. *******")
            utility.display_training_metrics(decision_tree_class, diabete_data)
            #utility.display_feature_importance(classifier, diabete_data)

        elif choice == 7:
            print("******* RandomForest *******")
            diabete_data.load_data()
            diabete_data.Preprocessing()
            diabete_data.data_split()
            classifier = RandomForestClassifier(random_state = 0, criterion='gini', max_depth=4)
            random_forest_class.model_fit(classifier, diabete_data)
            print("******* Neural Network. *******")
            utility.display_training_metrics(random_forest_class, diabete_data)
            #utility.display_feature_importance(classifier, diabete_data)


        elif choice == 8:
            print("***** LogisticRegression feature importance *******")
            diabete_data.load_important_data()
            diabete_data.data_split()
            lr_model = LogisticRegression(C=1, random_state=0, max_iter=1000)
            lr_class.model_fit(lr_model, diabete_data)
            print("******* Neural Network. *******")
            utility.display_training_metrics(lr_class, diabete_data)

            diabete_data.load_data()
            diabete_data.Preprocessing()
            diabete_data.data_split()
            lr_model = LogisticRegression(C=1, random_state=0, max_iter=1000)
            lr_class.model_fit(lr_model, diabete_data)
            utility.display_feature_importance(lr_model, diabete_data)
        else:
            print("Thank you for using machine learning system")
            exit()


if __name__ == "__main__":
    main()

