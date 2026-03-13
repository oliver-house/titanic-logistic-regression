# Logistic regression model to predict survival outcomes for passengers on the Titanic
# Datasets used and inspiration for this project are from the following Kaggle competition: https://www.kaggle.com/competitions/titanic
# Used for non-commercial, educational purposes under the competition rules

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.exceptions import ConvergenceWarning
import warnings
import os

def drop_uninformative_columns(df):
    """ Drops name, ticket, and cabin columns due to low predictive value """
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

def impute_training_ages(df):
    """ Imputes missing ages using median age for those with the same gender and class of ticket """
    df['Age'] = df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
    return df

def impute_testing_ages(df1, df2):
    """ Imputes missing ages in df2 using median age for those with the same gender and class of ticket in df1 """
    median_ages = df1.groupby(['Pclass', 'Sex'])['Age'].median()

    def impute_age(row):
        """ Imputes missing ages using medians from training data for a given row """
        if pd.isna(row['Age']):
            return median_ages.loc[row['Pclass'], row['Sex']]
        return row['Age']

    df2['Age'] = df2.apply(impute_age, axis=1)
    return df2

def impute_testing_fares(df1, df2):
    """ Imputes missing fare values as the median """
    df2['Fare'] = df2['Fare'].fillna(df1['Fare'].median())
    return df2

def impute_training_embarked(df):
    """ Imputes missing points of embarkation as the mode """
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df

def graphs(df):
    """ Saves three bar plots """
    os.makedirs('figures', exist_ok=True)
    df['Survived'].value_counts().plot(kind='bar')
    plt.title('Survival Totals')
    plt.xlabel('Outcome')
    plt.ylabel('Total')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join('figures', 'survival_totals.png'))
    plt.close()

    df.groupby('Sex')['Survived'].mean().plot(kind='bar')
    plt.title('Survival Rate by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join('figures', 'survival_by_gender.png'))
    plt.close()

    df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
    plt.title('Survival Rate by Class of Ticket')
    plt.xlabel('Class of Ticket')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join('figures', 'survival_by_class.png'))
    plt.close()

def exploratory_data_analysis(df, name='Dataset', show_summary=True, show_graphs=True):
    """ Displays summary statistics and graphs """
    if show_summary:
        print(f"{name}: ")
        print(df.describe())
        print('Missing values: ', df.isna().sum())
    if show_graphs and 'Survived' in df.columns:
        graphs(df)

def encode_categoricals(df):
    """ One-hot encodes categorical variables """
    return pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

def prepare_for_modelling(df1, df2):
    """ Reformats the data in preparation for logistic regression modelling """
    X_train = df1.drop(columns=['Survived'])
    y_train = df1['Survived'] 
    X_test = df2.reindex(columns=X_train.columns, fill_value=0)
    return (X_train, y_train, X_test)

def repeated_cross_validation(model, data, folds=5, repeats=10, rand_state=343, scoring='accuracy'):
    """ Performs repeated stratified K-fold cross-validation on the training data to test its accuracy """
    cross_val = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=rand_state)
    scores = cross_val_score(model, data[0], data[1], cv=cross_val, scoring=scoring)
    print(f"Mean accuracy percentage: {100*scores.mean():.0f} ± {100*scores.std():.0f}")

def predict(model, data):
    """ Performs logistic regression to obtain predictions """
    warnings.filterwarnings('error', category=ConvergenceWarning)
    try:
        model.fit(data[0], data[1])
    except ConvergenceWarning:
        print('Convergence too slow.')
    return model.predict(data[2])

if __name__ == '__main__':

    try:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
    except FileNotFoundError as fnfe:
        print(f"Error: {fnfe}")
        exit(1)

    # Print summary statistics and save bar plots

    exploratory_data_analysis(train, name='Training data')
    exploratory_data_analysis(test, name='Testing data')

    # Drop columns of limited predictive value

    drop_uninformative_columns(train)
    drop_uninformative_columns(test)

    # Impute missing values

    impute_training_ages(train)
    impute_testing_ages(train, test)
    impute_testing_fares(train, test)
    impute_training_embarked(train)

    train = encode_categoricals(train)
    test = encode_categoricals(test)

    # Reformat for logistic regression

    prepared = prepare_for_modelling(train, test)

    # Choose 5000 iterations for convergence to prevent early termination

    model = LogisticRegression(max_iter=5000)

    repeated_cross_validation(model, prepared)

    # Apply logistic regression model to full training data to obtain predictions for testing data

    predictions = predict(model, prepared)

    # Write predictions to CSV

    submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':predictions})
    submission.to_csv('predictions.csv', index=False)
