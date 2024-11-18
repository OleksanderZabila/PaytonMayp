import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn import tree
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(url, header=None, names=columns)

X = data.iloc[:, :4].values
labels = data.iloc[:, 4].values
le = LabelEncoder()
Y = le.fit_transform(labels)

random_state = 123
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=random_state)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(f"Точність GaussianNB: {gnb.score(X_test, y_test):.2f}")

test_sizes = np.arange(0.05, 1.0, 0.05)
errors = []
accuracies = []

for size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=random_state)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    errors.append((y_test != y_pred).sum())
    accuracies.append(gnb.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(test_sizes, errors, label='Помилки')
plt.plot(test_sizes, accuracies, label='Точність')
plt.xlabel('Розмір тестової вибірки')
plt.ylabel('Кількість помилок / Точність')
plt.legend()
plt.show()

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
print(f"Точність MultinomialNB: {mnb.score(X_test, y_test):.2f}")

cnb = ComplementNB()
cnb.fit(X_train, y_train)
print(f"Точність ComplementNB: {cnb.score(X_test, y_test):.2f}")

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
print(f"Точність BernoulliNB: {bnb.score(X_test, y_test):.2f}")

clf = tree.DecisionTreeClassifier(random_state=random_state)
clf.fit(X_train, y_train)
y_pred_tree = clf.predict(X_test)
print(f"Точність DecisionTreeClassifier: {clf.score(X_test, y_test):.2f}")
print(f"Кількість листків: {clf.get_n_leaves()}")
print(f"Глибина дерева: {clf.get_depth()}")

plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=columns[:4], class_names=le.classes_)
plt.show()

params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5]
}

for crit in params['criterion']:
    for split in params['splitter']:
        for depth in params['max_depth']:
            clf = tree.DecisionTreeClassifier(
                criterion=crit, splitter=split, max_depth=depth,
                min_samples_split=2, min_samples_leaf=1, random_state=random_state
            )
            clf.fit(X_train, y_train)
            print(f"Параметри: criterion={crit}, splitter={split}, max_depth={depth}")
            print(f"Точність: {clf.score(X_test, y_test):.2f}")
