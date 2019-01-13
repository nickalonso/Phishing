from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc

# Model selection 
def pipelines(X_train, X_test, y_train, y_test):

    # model pipelines
    pipe_logreg = Pipeline([('scl', preprocessing.MaxAbsScaler()),
                            ('clf', LogisticRegression(random_state=42))])

    pipe_decisiontree = Pipeline([('scl', preprocessing.MaxAbsScaler()),
                                  ('clf', tree.DecisionTreeClassifier(random_state=42))])

    # Transform vectors to be svm compatible
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    pipe_svm = Pipeline([('clf', svm.SVC(random_state=42))])

    # Set grid search parameters
    param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    param_range_fl = [1.0, 0.5, 0.1]

    grid_params_logreg = [{'clf__penalty': ['l1', 'l2'],
                       'clf__C': param_range_fl,
                       'clf__solver': ['liblinear']}]

    grid_params_decisiontree = [{
                       'clf__min_samples_leaf': param_range,
                       'clf__max_depth': param_range,}]

    grid_params_svm = [{'clf__kernel': ['linear', 'rbf'],
                        'clf__C': param_range}]

    # Construct grid searches
    jobs = -1

    gs_logreg = GridSearchCV(estimator=pipe_logreg,
                         param_grid=grid_params_logreg,
                         scoring='accuracy',
                         cv=10)

    gs_decisiontree = GridSearchCV(estimator=pipe_decisiontree,
                         param_grid=grid_params_decisiontree,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=jobs)

    gs_svm = GridSearchCV(estimator=pipe_svm,
                          param_grid=grid_params_svm,
                          scoring='accuracy',
                          cv=10,
                          n_jobs=jobs)

    # List of pipelines for ease of iteration
    grids = [gs_logreg, gs_svm, gs_decisiontree]
    # Dictionary of pipelines and classifier types for ease of reference
    grid_dict = {0: 'Logistic Regression', 1: 'Support Vector Machine', 2: 'Decision Tree'}
    # Fit the grid search objects
    print('Performing model optimizations...')
    best_acc = 0.0
    best_clf = 0
    best_gs = ''
    preds = []
    for idx, gs in enumerate(grids):
        print('\nModel: %s' % grid_dict[idx])
        # Fit grid search
        gs.fit(X_train, y_train)
        # Best params
        print('Best parameters: %s' % gs.best_params_)
        # Best training data accuracy
        print('Best training accuracy: %.3f' % gs.best_score_)
        # Predict on test data with best params
        y_pred = gs.predict(X_test)
        preds.append(y_pred)
        # Test data accuracy of model with best params
        print('Test set accuracy score for best parameters: %.3f ' % accuracy_score(y_test, y_pred))
        # Track best (highest test accuracy) model
        if accuracy_score(y_test, y_pred) > best_acc:
            best_acc = accuracy_score(y_test, y_pred)
            best_gs = gs
            best_clf = idx
    print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

    return preds

def roc_generator(y_test, pred1, pred3, pred4,):
    # Declare the number of classes and plot ROC curve
    num_classes = 2

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, pred1, pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr3 = dict()
    tpr3 = dict()
    roc_auc3 = dict()
    for i in range(num_classes):
        fpr3[i], tpr3[i], _ = roc_curve(y_test, pred3, pos_label=1)
        roc_auc3[i] = auc(fpr3[i], tpr3[i])

    fpr4 = dict()
    tpr4 = dict()
    roc_auc4 = dict()
    for i in range(num_classes):
        fpr4[i], tpr4[i], _ = roc_curve(y_test, pred4, pos_label=1)
        roc_auc4[i] = auc(fpr4[i], tpr4[i])

    fig = plt.figure(figsize=(15, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('#c1c5cc')
    # Major ticks every 0.05, minor ticks every 0.05
    major_ticks = np.arange(0.0, 1.0, 0.05)
    minor_ticks = np.arange(0.0, 1.0, 0.05)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='both')

    # Logistic Regression curve
    plt.plot(fpr[1], tpr[1], color='#4a50bf',
             lw=1, label='Logistic Regression (area = %0.4f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

    # Support Vector Machine
    plt.plot(fpr3[1], tpr3[1], color='#F6FF33',
             lw=1, label='SVM (area = %0.4f)' % roc_auc3[1])
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

    # Decision Tree curve
    plt.plot(fpr4[1], tpr4[1], color='#ff68f0',
             lw=1, label='Decision Tree (area = %0.4f)' % roc_auc4[1])
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristics')
    plt.legend(loc="lower right")
    plt.show()

def create_tokens(f):
    slashTokens = str(f.encode('utf-8')).split('/')  # splits URL by slash and gets tokens
    totalTokens = []
    for i in slashTokens:
        tokens = str(i).split('-')  # splits URL by dashes and gets tokens
        dotTokens = []
        for j in range(0, len(tokens)):
            temp = str(tokens[j]).split('.')  # splits url by dots and gets tokens
            dotTokens = dotTokens + temp
        totalTokens = totalTokens + tokens + dotTokens
    totalTokens = list(set(totalTokens))  # remove duplicates
    if 'com' in totalTokens:
        totalTokens.remove('com')  # pretty standard in a URL, will not be included in feature set
    return totalTokens

def main():
    load_data = '/Users/nickalonso/Documents/urldata.csv'
    csv = pd.read_csv(load_data)
    url_set = pd.DataFrame(csv)
    url_set = url_set.sample(frac=0.0001)
    y = url_set["label"]
    training_features = url_set["url"]

    # Tokenize URLs and define the tf-idf vectorizer for feature selection
    vectorizer = TfidfVectorizer(tokenizer=create_tokens)
    X = vectorizer.fit_transform(training_features)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Send vectors to model pipelines
    preds = pipelines(X_train, X_test, y_train, y_test)
    pred1, pred3, pred4 = preds[0], preds[1], preds[2]     # List of returned predictions from the method

    # Transform y vectors to be binary values ("good, bad" to 1, 0)
    y_test = y_test.values
    lb = preprocessing.LabelBinarizer()
    y_test = lb.fit_transform(y_test)
    pred1 = lb.fit_transform(pred1)
    pred3 = lb.fit_transform(pred3)
    pred4 = lb.fit_transform(pred4)

    # Generate ROC curve for model performance comparision
    roc_generator(y_test, pred1, pred3, pred4)

if __name__ == '__main__':
    main()
