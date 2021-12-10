from layersdk import dataset, model, Layer, File, Dataset, SQL, assert_unique, \
    assert_not_null, assert_valid_values, assert_true
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data_file = 'titanic.csv'


def check_for_fare(df):
    return df.Fare.max() <= 520 and df.Fare.min() >= 0


# Create dataset from local file
@assert_true(check_for_fare)
@assert_not_null('Name')
@assert_unique('PassengerId')
@assert_valid_values('Sex', ['male', 'female'])
@dataset('raw_passengers', depends=[File(data_file)])
def read_and_clean_dataset():
    df = pd.read_csv(data_file)
    layer.log(f"Total passengers: {len(df)}")
    return df


# Create dataset with sql
# @dataset('raw_passengers')
# def read_and_clean_dataset():
#     return SQL(f'select * from titanic')

# Create dataset from an integration
# @dataset('raw_passengers')
# def read_and_clean_dataset():
#     return layersdk.Datasource('layer-public-datasets', 'titanic')


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


@dataset('features', depends=[Dataset('raw_passengers')])
def extract_features():
    df = layer.get_dataset("raw_passengers").to_pandas()

    df['Sex'] = df['Sex'].apply(clean_sex)
    df['Age'] = df[['Age', 'Pclass']].apply(clean_age, axis=1)

    df = df.drop(["PassengerId", "Name", "Cabin", "Ticket", "Embarked"], axis=1)

    layer.log(f'Features: {list(df.columns)}')
    layer.log(f'Total Count: {len(df)}')
    return df


def dummy_passengers():
    # Based on passenger 2 (high passenger class female)
    passenger2 = {'PassengerId': 2,
                  'Pclass': 1,
                  'Name': ' Mrs. John',
                  'Sex': 'female',
                  'Age': 38.0,
                  'SibSp': 1,
                  'Parch': 0,
                  'Ticket': 'PC 17599',
                  'Fare': 71.2833,
                  'Embarked': 'C'}

    return passenger2


def get_passenger_features(df):
    df['Sex'] = df['Sex'].apply(clean_sex)
    df['Age'] = df[['Age', 'Pclass']].apply(clean_age, axis=1)
    return df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]


def test_survival_probability(model):
    # Changing gender from female to male should decrease survival probability.
    p2 = dummy_passengers()

    # Get original survival probability of passenger 2
    test_df = pd.DataFrame.from_dict([p2], orient='columns')
    X = get_passenger_features(test_df)
    p2_prob = model.predict_proba(X)[0][1]  # 0.99

    # Change gender from female to male
    p2_male = p2.copy()
    p2_male['Sex'] = 'male'
    test_df = pd.DataFrame.from_dict([p2_male], orient='columns')
    X = get_passenger_features(test_df)
    p2_male_prob = model.predict_proba(X)[0][1]  # 0.53

    return p2_male_prob < p2_prob


@assert_true(test_survival_probability)
@model(name='survival_model', depends=[Dataset('features')], fabric='f-small')
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
# df = read_and_clean_dataset()

# ++ To run the whole project on Layer Infra
layer.run([read_and_clean_dataset, extract_features, train])

# ++ To train model on Layer infra
# layer.run([train])

# ++ To debug the code locally, just call the function:
# train()
# extract_features()

# read_and_clean_dataset()
# train()
