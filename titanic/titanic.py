from layer import dataset, model, Layer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


@dataset('raw_passengers')
def read_and_clean_dataset():
    df = pd.read_csv("titanic.csv")
    layer.log(f"Total passengers: {len(df)}")
    return df


def clean_sex(sex):
    result = 0
    if sex == "female":
        result = 0
    elif sex == "male":
        result = 1
    return result


def clean_age(data):
    age = data[0]
    pclass = data[1]
    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


@dataset('features')
def extract_features():
    df = layer.get_dataset("raw_passengers").to_pandas()

    df['Sex'] = df['Sex'].apply(clean_sex)
    df['Age'] = df[['Age', 'Pclass']].apply(clean_age, axis=1)

    df = df.drop(["PassengerId", "Name", "Cabin", "Ticket", "Embarked"], axis=1)

    layer.log(f'Features: {list(df.columns)}')
    layer.log(f'Total Count: {len(df)}')
    return df


@model(name='survival_model')
def train():
    df = layer.get_dataset("features").to_pandas()
    layer.log(f"Training data count: {len(df)}")

    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    layer.log_metric("accuracy", f'{acc:.4f}')
    return random_forest


# ++ init Layer
layer = Layer(project_name="ltv_project", environment='requirements.txt')


# ++ To run the whole project on Layer Infra
layer.run([read_and_clean_dataset, extract_features, train])
# layer.run([build_dummy, train, read_and_clean_dataset, extract_features])

# ++ To train model on Layer infra
# layer.run([train])

# ++ To debug the code locally, just call the function:
# train()
# extract_features()

# read_and_clean_dataset()
# train()