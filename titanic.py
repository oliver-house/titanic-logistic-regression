"""Logistic regression model to predict survival outcomes for passengers on the Titanic"""

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

def impute_testing_ages(df1, df2):
    """ Imputes missing ages in df2 using median age for those with the same gender and class of ticket in df1 """
    median_ages = df1.groupby(['Pclass', 'Sex'])['Age'].median().reset_index(name='median_age')
    df2 = df2.merge(median_ages, on=['Pclass', 'Sex'], how='left')
    df2['Age'] = df2['Age'].fillna(df2['median_age'])
    return df2.drop(columns=['median_age'])

def impute_testing_fares(df1, df2):
    """ Imputes missing fare values as the median """
    df2['Fare'] = df2['Fare'].fillna(df1['Fare'].median())

def impute_training_embarked(df):
    """ Imputes missing points of embarkation as the mode """
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

def graphs(df):
    """ Saves three bar plots """
    os.makedirs('outputs', exist_ok=True)
    df['Survived'].value_counts().plot(kind='bar')
    plt.title('Survival Totals')
    plt.xlabel('Outcome')
    plt.ylabel('Total')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join('outputs', 'survival_totals.png'))
    plt.close()

    df.groupby('Sex')['Survived'].mean().plot(kind='bar')
    plt.title('Survival Rate by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join('outputs', 'survival_by_gender.png'))
    plt.close()

    df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
    plt.title('Survival Rate by Class of Ticket')
    plt.xlabel('Class of Ticket')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join('outputs', 'survival_by_class.png'))
    plt.close()


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
    """ Performs repeated stratified K-fold cross-validation on the training data to estimate the model's accuracy """
    cross_val = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=rand_state)
    scores = cross_val_score(model, data[0], data[1], cv=cross_val, scoring=scoring)
    print(f"Mean accuracy percentage: {100*scores.mean():.0f} ± {100*scores.std():.0f}")

def predict(model, data):
    """ Fits the model and returns predictions for the test set """
    warnings.filterwarnings('error', category=ConvergenceWarning)
    try:
        model.fit(data[0], data[1])
    except ConvergenceWarning:
        print('Convergence too slow.')
        exit(1)
    return model.predict(data[2])

if __name__ == '__main__':
    try:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
    except FileNotFoundError as fnfe:
        print(f"Error: {fnfe}")
        exit(1)

    graphs(train)
    drop_uninformative_columns(train)
    drop_uninformative_columns(test)
    impute_training_ages(train)
    test = impute_testing_ages(train, test)
    impute_testing_fares(train, test)
    impute_training_embarked(train)
    train = encode_categoricals(train)
    test = encode_categoricals(test)
    prepared = prepare_for_modelling(train, test)
    model = LogisticRegression(max_iter=5000) # Choose 5000 iterations for convergence to prevent early termination
    repeated_cross_validation(model, prepared)
    predictions = predict(model, prepared)
    submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':predictions})
    submission.to_csv(os.path.join('outputs', 'predictions.csv'), index=False)