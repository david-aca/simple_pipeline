import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder

class Preprocessor:
    def __init__(self, prepocessor_type):
        scaler_dict = {'minmax': MinMaxScaler(),
                       'norm': StandardScaler()}
        self.scaler = scaler_dict[prepocessor_type]()

    def fit(self, X):
        if 'target' in X.columns:
            y = X['target']
            self.X_numeric = X.select_dtypes(include=['int', 'float'])
            XX = self.scaler.fit(self.X_numeric)
            self.X_new = self.X_numeric.drop('target', axis=1)
        return self.X_new

    def transform(self, X):
        numeric_columns =self.X_new.select_dtypes(include=['int', 'float']).columns
        numeric_means = self.X_new[numeric_columns].mean()
        self.X_new[numeric_columns] = self.X_new[numeric_columns].fillna(numeric_means)
        cat_columns = self.X_new.select_dtypes(include=['object']).columns.tolist()
        if len(cat_columns) > 0:
            one_encode = OneHotEncoder(sparse = False)
            one_encode_array = one_encode.fit_transform(X_new[cat_columns])
            one_encoded_df = pd.DataFrame(one_encode_array, columns=pd.get_dummies(X_new[cat_columns]).keys())
            XX= self.X_new.drop(columns=cat_columns)
            X= pd.concat([XX, one_encoded_df], axis=1)



        scaler1 = self.scaler.fit(self.X_new)
        result = scaler1.transform(self.X_new)
        X = pd.DataFrame(result, columns=(self.X_new.columns))
        return X



