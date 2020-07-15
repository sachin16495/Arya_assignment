The Problem statement was to a binary classification with proper preprocessing and EDA we started getting an understanding of data and found out all the features, not from any use cases. The data is somehow similar to sensor data we stared of checking the missing value in the data it was not there.

Then we move to the next step which visualizing the correlation in the data since the data set had large feature so we decided to go with Correlation heat which gives us a great result and help us to remove multicollinearity.
In the next, we have to visualize the data the only way to visualize the data when you have high features is with PCA on visualizing post applying PCA we found out that the data can be linearly separable.
After that, we start we extra tree Classifier to find out which feature is more important the extra tree classifier is based on the Information gain on applying extra tree classifier the features get reduced to 34.

We will do our train test split after that we are trying to visualize the decision boundary for our dataset which gives information about how the features get distributed.
Post this step we are trying to build our first model since while visualizing with PCA we found the data can be separated linearly in the 2D plane so we try to train our logistics regression model and after a train with the metrics, we found that the result was good for the simple model like logistic regression.
Then we move to the next model since logistic regression is the linear model we try to build the model which can separate the data in a plane so come up the support vector but unfortunately, it didn’t perform well

Then we move to XGBoost model and it performs remarkably well in our case, We generated our test result with XGBoast only we also Gaussian Naive Biase because after PCA visualization there was a hintch that it may for Gaussian distribution but was wrong finally we tried Random Forest the result was good but Random Forest is more prone to overfitting the data give a low accurate result on shuffling the testing data
