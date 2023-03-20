from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


def preprocess(df):
    # Extract relevant columns
    categorical_columns = ["Sex", "Embarked"]
    numerical_columns = ["Age", "Fare", "Pclass", "Parch"]
    feat_columns = categorical_columns + numerical_columns

    # Ordinal encode categorical columns
    df[categorical_columns] = OrdinalEncoder().fit_transform(df[categorical_columns])

    # Impute/drop missing values
    df.loc[:, "Age"].fillna(df["Age"].mean(), inplace=True)
    df = df.dropna()
    
    X_df = df.loc[:, feat_columns]
    y_df = df.loc[:, "Survived"]

    return X_df, y_df


def get_titanic(path="raw_data/titanic/titanic_train.csv"):
    # Load the data
    df = pd.read_csv(path)

    # Split df into train and test set
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    X_train, y_train = preprocess(df_train)
    X_test, y_test = preprocess(df_test)
    return X_train, X_test, y_train, y_test