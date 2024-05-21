Project Description: Wine Quality Prediction using Random Forest

Overview:
This project aims to predict the quality of wine based on various chemical properties using the Random Forest algorithm. The dataset used in this project contains information about red and white wine samples, including their chemical composition and quality ratings.

Dataset:
The dataset, named "winequality_combined.csv", consists of the following features:
- wine_type: Indicates whether the wine is red (r) or white (w).
- fixed acidity: The amount of fixed acids in the wine.
- volatile acidity: The amount of volatile acids in the wine.
- citric acid: The amount of citric acid in the wine.
- residual sugar: The amount of residual sugar in the wine.
- chlorides: The amount of chlorides in the wine.
- free sulfur dioxide: The amount of free sulfur dioxide in the wine.
- total sulfur dioxide: The total amount of sulfur dioxide in the wine.
- density: The density of the wine.
- pH: The pH level of the wine.
- sulphates: The amount of sulphates in the wine.
- alcohol: The alcohol content of the wine.
- quality: The quality rating of the wine (ranges from 3 to 9).

Data Preprocessing:
The project performs several data preprocessing steps:
1. The target variable 'quality' is separated from the feature variables.
2. The categorical feature 'wine_type' is encoded using one-hot encoding.
3. The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn.
4. The feature variables are scaled using the `StandardScaler` from scikit-learn to standardize the data.

Model 1 Training and Evaluation:
1. A Random Forest classifier was initialized with 200 estimators and a random state of 10.
2. The model is trained on the scaled training data using the `fit` method.
3. Predictions are made on the scaled testing data using the `predict` method.
4. The confusion matrix and accuracy score are calculated to evaluate the model's performance.
5. The classification report is generated to provide detailed metrics for each quality class.

Model 1 Optimization:

In an attempt to improve the model's performance, the number of estimators in the Random Forest classifier is increased to 1000. The model is retrained with the updated parameters, and the evaluation metrics are recalculated.

Model 1 Results:

The initial Random Forest model with 200 estimators achieves an accuracy score of approximately 0.67. The confusion matrix and classification report provide insights into the model's performance for each quality class. The model performs well in predicting quality ratings of 5, 6, and 7, but struggles with the less frequent classes (3, 4, 8, and 9).

After increasing the number of estimators to 1000, the accuracy score slightly improves to approximately 0.68. However, the model still has difficulty predicting the less frequent quality classes accurately.

Model 2:

Since increasing the number of estimators in the previous model did not achieve a target accuracy score of .75 or higher, a different approach was taken to group the wine into binary classes: good (rating of 7 or higher) and bad (rating less than 7).

A second Random Forests model was instantiated with the binary classifications using 200 estimators. An accuracy score of .89 was attained with the binary model. 

Feature Importance:

The project also explores the feature importance of the Random Forest model. The `feature_importances_` attribute is used to retrieve the importance scores for each feature. The features are then sorted based on their importance and visualized using a horizontal bar plot.

Conclusion:

This project demonstrates the application of the Random Forest algorithm for predicting wine quality based on chemical properties. The first model achieved a poor accuracy score due to imbalanced classes. After recategorizing the data into binary classes for wine quality, the second model performed substantially better.

The project provides a solid foundation for understanding the process of training and evaluating a Random Forest model for wine quality prediction.

Sources:

https://www.kaggle.com/code/margijiyani1/notebook093682bdb/input?select=winequality_white.csv

https://www.dataquest.io/blog/tutorial-add-column-pandas-dataframe-based-on-if-else-condition/

Link to Presentation: https://docs.google.com/presentation/d/1D_0GlKvrI3_fMi7zgD_YqZJf3nEujUL4CcSmm1mMUkM/edit#slide=id.p
