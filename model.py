from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


def train_decision_tree_classifier(x_train, y_train, criterion="gini", max_depth=None, min_samples_split=2,
                                   random_state=42):
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                   random_state=random_state)
    model.fit(x_train, y_train)
    return model


def train_knn(x_train, y_train, n_neighbors=5, metric='minkowski', weights='uniform'):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, n_estimators=100, max_depth=None, criterion='gini', random_state=42):
    model = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state)
    model.fit(x_train, y_train)
    return model


def train_svc(x_train, y_train, kernel='rbf', c=1.0, gamma='scale', random_state=None):
    model = SVC(kernel=kernel, C=c, gamma=gamma, random_state=random_state, probability=True)
    model.fit(x_train, y_train)
    return model


def train_logistic_regression(x_train, y_train, c=1.0, solver='lbfgs', penalty='l2', random_state=42):
    model = LogisticRegression(solver=solver, C=c, penalty=penalty, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def train_linear_regression(x_train, y_train, fit_intercept=True):
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(x_train, y_train)
    return model


def train_decision_tree_regressor(x_train, y_train, criterion="squared_error", max_depth=None,
                                  min_samples_split=2, random_state=42):
    model = DecisionTreeRegressor(criterion=criterion, min_samples_split=min_samples_split, max_depth=max_depth,
                                  random_state=random_state)
    model.fit(x_train, y_train)
    return model


def train_random_forest_regressor(x_train, y_train, n_estimators=100, max_depth=None, criterion="squared_error",
                                  random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
                                  random_state=random_state)
    model.fit(x_train, y_train)
    return model
