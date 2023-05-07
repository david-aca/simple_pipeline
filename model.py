from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

class Model:
    def __init__(self, model_type):
        classifier_dict = {'svc': SVC(kernel="rbf", C=1, gamma=2, probability=True),
                           'randomForest': RandomForestClassifier(n_estimators=45, oob_score=True, max_depth=10),
                           'baggingSVC': BaggingClassifier(SVC(kernel="linear", C=1, probability=True), n_estimators=21, oob_score=True),
                           'DecisionTreeClassifier': DecisionTreeClassifier(ccp_alpha = 0.0001, max_depth=100)}

        self.classifier = classifier_dict[model_type]()

    def fit(self, X, y):
        #X, y are train data
        self.classifier.fit(X, y)

    def predict_proba(self, X):

        y_pred_proba = self.classifier.predict_proba(X)

        return y_pred_proba
