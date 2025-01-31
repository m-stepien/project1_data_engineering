from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd

param_grid = {
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    },
    'DecisionTreeRegressor': {
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 10, None],
        'criterion': ['gini', 'entropy']
    },
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 10, None],
        'criterion': ['squared_error', 'absolute_error']
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 10, 15],
        'metric': ['minkowski', 'euclidean', 'manhattan'],
        'weights': ['uniform', 'distance']
    },
    'SVC': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf', 'poly']
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l1', 'l2']
    },
    'LinearRegression': {
        'fit_intercept': [True, False]
    }
}


def grid_search(model, x_train, y_train, scoring='accuracy'):
    model_name = type(model).__name__
    cv = GridSearchCV(estimator=model, param_grid=param_grid[model_name], cv=5, scoring=scoring, n_jobs=-1, return_train_score=True)
    cv.fit(x_train, y_train)
    results = pd.DataFrame(cv.cv_results_)
    results = results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    print(results)
    print(f"Best {model_name}: {cv.best_params_}")
    print(f"Best score: {cv.best_score_}")

    return cv.best_estimator_, results


def random_search(model, x_train, y_train, scoring='accuracy', n_iter=3):
    model_name = type(model).__name__
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid[model_name],
        n_iter=n_iter,
        cv=2,
        scoring=scoring,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )
    rs.fit(x_train, y_train)
    results = pd.DataFrame(rs.cv_results_)
    results = results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    print(results)
    print(f"Best {model_name}: {rs.best_params_}")
    print(f"Best score: {rs.best_score_}")

    return rs.best_estimator_, results
