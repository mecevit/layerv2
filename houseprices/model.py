from layer import Layer, dataset, model
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class HousePricePredictor:
    def __init__(self, layer: Layer, epochs):
        self.layer = layer
        self.epochs = epochs

    @model('ltv_model')
    def train(self):
        df_train = self.layer.get_dataset("train").to_pandas()
        for epoch in range(self.epochs):
            self.layer.log(f"Epoch {epoch + 1}/{self.epochs}")


@dataset('train')
def train():
    df = pd.read_csv("train.csv")
    layer.log(f"Total train size: {len(df)}")
    return df


layer = Layer(project_name="house_prices")
layer.run([train, HousePricePredictor(layer, 5).train])
