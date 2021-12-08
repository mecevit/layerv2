# Original Repo:
# https://github.com/tiwari91/Housing-Prices

from layer import Layer, dataset, model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn import decomposition

class App():
    def __init__(self, file_train, file_test, layer:Layer):
        self.file_train = file_train
        self.file_test = file_test
        self.layer = layer

    @dataset("train")
    def create_training_data(self):
        df = pd.read_csv(self.file_train, index_col=0)
        return df

    @dataset("test")
    def create_test_data(self):
        return pd.read_csv(self.file_test, index_col=0)

    def _normalizeData(self, Numeric_columns):
        # Function to normalize
        means = self._df.loc[:, Numeric_columns].mean()
        stdev = self._df.loc[:, Numeric_columns].std()
        self._df.loc[:, Numeric_columns] = (self._df.loc[:,
                                            Numeric_columns] - means) / stdev

        index_train = self._df.loc[self._train_df.index]
        index_test = self._df.loc[self._test_df.index]

        self._xTrain = index_train.values
        self._xTest = index_test.values

        self._df['LotArea'] = np.log(self._df['LotArea'])
        self._df['LotFrontage'] = np.log(self._df['LotFrontage'])

    def _removeSkewness(self):
        # Store target variable and remove skewness
        target = self._train_df['SalePrice']
        plt.hist(target)
        plt.show()
        del self._train_df['SalePrice']

        self._yTrain = np.log(target)
        plt.hist(self._yTrain)
        plt.xlabel('SalePrice')
        plt.show()

    def _dummyCreate(self):
        # Create dummy variables for the categorical features and handle the missing values
        self._df = pd.get_dummies(self._alldf)
        self._df.isnull().sum().sort_values(ascending=False)
        self._df = self._df.fillna(self._df.mean())

    """
    """

    def _pcaLassoRegr(self):
        pca = decomposition.PCA()
        pca.fit(self._xTrain)

        fig = plt.figure(1, figsize=(4, 3))

        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(pca.explained_variance_, linewidth=2)
        plt.axis('tight')
        plt.xlabel('n_components')
        plt.ylabel('explained_variance_')
        plt.show()

        train_pca = pca.transform(self._xTrain)
        test_pca = pca.transform(self._xTest)

        lassoregr = LassoCV(
            alphas=[0.1, 0.001, 0.0001, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                    12]).fit(train_pca, self._yTrain)
        rmse = np.sqrt(-cross_val_score(lassoregr, train_pca, self._yTrain,
                                        scoring="neg_mean_squared_error",
                                        cv=5)).mean()
        layer.log_metric("rmse", rmse)
        y_lasso = lassoregr.predict(self._xTest)

        return y_lasso

    @model("house_price_predictor")
    def train(self):
        self._train_df = layer.get_dataset("train").to_pandas()
        self._test_df = layer.get_dataset("test").to_pandas()

        # Remove skewness
        self._removeSkewness()

        # Concatenates the data
        self._alldf = pd.concat((self._train_df, self._test_df), axis=0)

        # Creates dummy variables
        self._dummyCreate()

        # Retrieve all numeric features
        numeric_columns = self._alldf.columns[self._alldf.dtypes != 'object']

        # Normalize the data set
        self._normalizeData(numeric_columns)

        # Using PCA and LassoReggression
        y_final = self._pcaLassoRegr()


layer = Layer(project_name="house_prices")

app = App('train.csv', 'test.csv', layer)
layer.run(
    [
        app.create_training_data,
        app.create_test_data,
        app.train,
    ]
)
