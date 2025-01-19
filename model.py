from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,  RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


def train_decision_tree_classifier(x_train, y_train, criterion="gini", max_depth=None):
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    model.fit(x_train, y_train)
    return model


def train_knn(x_train, y_train, n_neighbors, metric):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    model.fit(x_train, y_train)
    return model


def train_random_forest(x_train, y_train, n_estimators, max_depth, random_state):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def train_svc(x_train, y_train, kernel, c, gamma, random_state):
    model = SVC(kernel=kernel, C=c, gamma=gamma, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def train_logistic_regression(x_train, y_train, multi_class, solver):
    model = LogisticRegression(multi_class=multi_class, solver=solver)
    model.fit(x_train, y_train)
    return model


def train_linear_regression(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def train_decision_tree_regressor(x_train, y_train, max_depth, random_state):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def train_random_forest_regressor(x_train, y_train, n_estimators, random_state):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def predict(x_test, model):
    return model.predict(x_test)