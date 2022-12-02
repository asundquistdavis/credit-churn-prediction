# Credit Card Churn

### Analyzing and Predicting Credit Card Churn

##### Authors:
* Bryan Groves (GrovesB) - KNN analysis and lead Tableau design
* Nathan Johnson (ntjohn3551) - Logistic Regression analysis
* Jacob McManaman (Jacob-McM) - Neural Network analysis
* Andrew Sundquist (asundquistdavis) - Lead Web Hosting and Random Forest Classifier analysis
* John Torgerson (JohnTorgerson) - Lead Web design
---

### Overview:
This project aims to predict credit card user churn by analysing a data set containing over 25,000 credit cards users. The data set contains both demographical and usage - months inactive, credit limit, number of transactions - features as well as their active status. The original data indicates existing and non-existing users however, the models created are used as a way to predict which types of users are more likeing to close their accounts in the future.

While the data set in this project is used to predict churn this same type of analysis can be used to predict churn patterns on other data sets.

---


### Guide to Repo Contents:
* In Folder `Analysis`: this folder contains python files and notebooks performing the analysis for the project
    1. `etl_workflow.ipynb` is a notebook that cleans the data, explores its basic features and prepares it for further analysis
    1. `KNN.ipynb` is a jupyter notebook for modeling using k-nearest neighbors
    1. `Logistic_Regression.ipynb` is a jupyter notebook for modeling using logistic regression
    1. `random-forest-classifier.ipynb` is a jupyter notebook for modeling using random forest classifier
    1. `Dimensionality-Reduction.ipynb` is a jupyter notebook that explores reducing the data set 
    1. `mechlearn.py` is a custom python module that defines functions to help with the analysis
    1. `model.py` is a python file that produces an rfc model (and the models accompaning scalar) and saves it as .pkl file for use in app.py
* In Folder `Models`: this folder contains scikit-learn model objects saved as .pkl files to be used in app.py
* In Folder `Scalers`:
* In Folder `Outputs`:
    1. `KNeighborsClassifier_ROC.csv` is a csv table for the K-nearest neighbors ROC curve data
    2. `LogisticRegression_ROC.csv` is a csv table for the logistic regression ROC curve data
    3. `RandomForestClassifier_ROC.csv` is a csv table for the random forest classifier ROC curve data
    4. `SVC_ROC.csv` is a csv table for the C-support vector classifier ROC curve data
* In Folder `Resources`: this folder holds the data used to train and test mechine learning models as well as assets used in the project
    1. `bank_churn_sketch.jpg` is an image file showing a rough sketch of our model page
    2. `BankChurners.csv` is the raw data of banking customers
    3. `churn.ico` is an icon file for loading our favicon
    4. `churn.png` is an image file for our project logo
    5. `clean_churn_db.csv` is the output file of cleaned data of banking customers
    6. `X.csv` is the training set stored as a data file
    7. `y.csv` is the test set stored as a data file
* In Folder `static/css`: this folder holds all stylesheets used in the website
    1. `reset.css` is a style sheet used to clear and reset styles to standardized starting point
    2. `style.css` is a style sheet used to customize the html pages
* In Folder templates: this folder holds all .html files for each page of the website
    1. `about.html` is an html file for the about webpage
    1. `index.html` is an html file for the home landing page
    1. `predict.html` is an html file for storing a template we plan to host on a heroku app
    * In Folder `models`: each file in this folder is an .html file for a page on the website corespounding to a different model used in the project
        1. `knn.html` is an html file for the k-nearest neighbors webpage
        2. `logistic_regression.html` is an html file for the logistic regression webpage
        3. `neural_network` is an html file for the neural network webpage
        4. `random_forest` is an html file for the random forest webpage
* `app.py` is the python file that runs the flask server for the website
* `README.md` is the file you're currently viewing that summarizes this repo
* `Procfile` is a text file that indicates how Heroku should host the website
* `requirements.txt` is a text file that indicates to Heroku the enviroment needed to serve the website
* `runtime.txt` is a text file that indicates runtime variables to Heroku
---

### Observations:
* 
---

### Conclusions:
* 
---

### Credits and Special Thanks
* whoever
* University of Minnesota in partnership with 2U inc
* Instructor Dominic LaBella
* TA Colin
* TA Nick
* TA Chris