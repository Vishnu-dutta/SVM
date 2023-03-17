import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()
print(digits.target)
print(dir(digits))
print(digits.target_names)

df = pd.DataFrame(digits.data, digits.target)
print(df.head())

df['target'] = digits.target
print(df.head(20))

X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis='columns'), df.target, test_size=0.3)

'Using RBF kernel'
# rbf_model = SVC(kernel='rbf')
#
# print(len(X_train))
# print(len(X_test))
#
# rbf_model.fit(X_train, y_train)
# print(rbf_model.score(X_train, y_train))

'Using Linear Kernel'

linear_model = SVC(kernel='linear')
linear_model.fit(X_train, y_train)
print(linear_model.score(X_test, y_test))


