from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

parameter = {
    'min_samples_split': [1, 2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}


def grid_search(model, x_train, y_train):
    cv = GridSearchCV(estimator=model, param_grid=parameter, cv=5)
    cv.fit(x_train, y_train)


def random_search(model, x_train, y_train):
    rs = RandomizedSearchCV(estimator=model, param_distributions=parameter, n_iter=50, cv=5,
                            verbose=2, random_state=42, n_jobs=-1)
    rs.fit(x_train, y_train)
