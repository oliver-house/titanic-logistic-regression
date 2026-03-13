# Titanic Survival Predictions Using Machine Learning

In this project, I take a sample of over 800 passengers on the Titanic (`train.csv`), including their age, gender, class of ticket, and whether or not they survived. I also take a different sample of over 400 passengers (`test.csv`), with the same data provided, except for whether or not they survived. I apply logistic regression to the data in `train.csv` to predict survival outcomes for passengers in `test.csv`.

## Methodology

I chose logistic regression as an initial model since it is a relatively simple machine learning method suitable for binary classification tasks. I also chose to keep the feature engineering simple to focus on building a clean baseline model. No hyperparameter tuning was carried out and only a small number of raw features were used.

## Validation

I use repeated stratified K-fold cross-validation to evaluate the model by randomly dividing the training data into 5 groups, where each group has the same survival rate as the full training data. For each one of these groups, I train the model on the remaining 4 groups to derive predictions for the specified group. I then see how accurately the model predicted survival in the specified group. I repeat this process across all 5 groups, and repeat the entire cycle 10 times, yielding a sample of 50 accuracy scores with mean 80% (±3%).

## Results

Once complete, the program saves a CSV file (`outputs/predictions.csv`) with predictions for whether they survived (1) or died (0). Via submission to Kaggle, I can confirm that the model achieved an accuracy of 0.76076 (76%) for predictions on the test set `test.csv`, which is broadly consistent with the cross-validation estimate of 0.80 ± 0.03. 

## Scope for further development

More powerful models will likely yield more accurate predictions. More intensive feature engineering, such as extracting titles from the 'Name' columns, could yield further insights. This project serves as a foundation for more advanced modelling work.

## Acknowledgements

The datasets and inspiration for the project came from the Kaggle competition [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). Both datasets (`train.csv`, `test.csv`) are used here for non-commercial, educational purposes under their [competition rules](https://www.kaggle.com/competitions/titanic/rules#7-competition-data).
