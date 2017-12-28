#from eda import final_df
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

def final_df(f):
    orig_df = pd.read_csv(f)
    col_dropped_df = orig_df.drop(['Name', 'Cabin', 'Ticket'], axis=1)
    embarked_dummies = pd.get_dummies(col_dropped_df['Embarked'], drop_first=True, dummy_na=True)
    sex_dummies = pd.get_dummies(col_dropped_df['Sex'], drop_first=True, dummy_na=False)
    drop_again = col_dropped_df.drop(['Embarked', 'Sex'], axis=1)
    dummified_df = pd.concat((drop_again, embarked_dummies, sex_dummies), axis=1)
    dummified_df['null_age'] = dummified_df['Age'].isnull() == True
    dummified_df.loc[dummified_df['null_age'] == True, 'Age'] = dummified_df['Age'].median()
    final_df = dummified_df
    return final_df

def get_x_y(filename):
    df = final_df(filename)
    y = df.pop('Survived').values
    X = df.values
    return X, y

X_train, y_train = get_x_y('train.csv')
X_test = final_df('test.csv')
X_test.info()
X_test.loc[X_test['Fare'].isnull() == True, 'Fare'] = X_test['Fare'].median()

models = {
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'LogisticRegression': LogisticRegression()
}

params = {
    'RandomForestClassifier': { 'n_estimators': [16, 32], 'max_features': [2, 3, 4, 5, 6, 7, 8]}, #consider min_sample_leaf
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.1, 0.3, 0.5, 0.8, 1.0], 'subsample': [0.5], 'max_depth': [None, 5, 10, 15]},
    'LogisticRegression': {'C': [0.001, 0.01, 0.1, 1., 10., 100.]}
}

class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.best_estimator = {}

    def fit(self, X, y, cv=5, n_jobs=1, verbose=1, scoring=None, refit=True):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs
            self.best_estimator[key] = gs.best_estimator_
            print gs.best_score_
     #
    #  def score_summary(self):
    #      for key in self.keys:
    #          print self.best_estimator[key].best_score_

helper = EstimatorSelectionHelper(models, params)
helper.fit(X_train, y_train, scoring='f1', n_jobs=-1)

model = helper.best_estimator['RandomForestClassifier']
results = model.predict(X_test)
results_df = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Survived': results})
results_df.set_index(['PassengerId'], inplace=True)
results_csv = results_df.to_csv('/Users/hslord/kaggle/titanic/results.csv')
