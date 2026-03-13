# Titanic Survival Predictions Using Machine Learning

I apply logistic regression to the Titanic training data (`train.csv`) to predict survival outcomes for passengers in `test.csv`.

## Methodology

I chose logistic regression as an initial model since it is a relatively simple method suitable for binary classification tasks. No hyperparameter tuning was carried out and only a small number of raw features were used.

## Results

Predictions are saved to `outputs/predictions.csv` (1 = survived, 0 = died). Using repeated stratified K-fold cross-validation (5 folds, 10 repeats), the model achieves a mean accuracy of 80% (±3%) on `train.csv`. Submission to Kaggle gives an accuracy of 76% on `test.csv`, broadly consistent with the cross-validation estimate.

## Scope for further development

More powerful models will likely yield more accurate predictions. More intensive feature engineering, such as extracting titles from the 'Name' column, could also improve performance.

## Usage

Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/titanic/data) and place them in the project root, then run `titanic.py`.

## Acknowledgements

The datasets and inspiration for the project came from the Kaggle competition [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). Both datasets (`train.csv`, `test.csv`) are used here for non-commercial, educational purposes under their [competition rules](https://www.kaggle.com/competitions/titanic/rules#7-competition-data).
