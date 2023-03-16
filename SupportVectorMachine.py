import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


iris = load_iris()
print(iris.feature_names)
print(iris.target_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df[df.target == 1])
print(df[df.target == 2])

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

print(df[45:55])

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green')
# plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue')


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue')
plt.show()


X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
len(X_train)
len(X_test)

reg = SVC()
reg.fit(X_train,y_train)

print(reg.score(X_test, y_test))
print(reg.predict([[4.8,3.0,1.5,0.3]]))

'Regularization (C)'
reg_C = SVC(C=1)
reg_C.fit(X_train,y_train)
print(reg.score(X_test,y_test))

reg_C = SVC(C=10)
reg_C.fit(X_train,y_train)
print(reg.score(X_test,y_test))

'Gamma'
reg_G = SVC(gamma=10)
reg_G.fit(X_train,y_train)
print(reg.predict(X_test,y_test))

'Kernel'
reg_linear_kernel = SVC(kernel='linear')
reg_linear_kernel.fit(X_train, y_train)
print(reg_linear_kernel.predict(X_test, y_test))

