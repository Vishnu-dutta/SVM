from sklearn.datasets import load_digits
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


digits = load_digits()
print(dir(digits))

'''
creating Dataframe using the dataset of Load_digits and further appending 
the target column into it. Data is row and target is column. 
In addition target is appended to the columns namely
'''
df = pd.DataFrame(digits.data, digits.target)
df['target'] = digits.target
print(df.head())

reg_normal = SVC(kernel='linear')
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=2)

reg_normal.fit(X_train, y_train)
print(reg_normal.score(X_test, y_test))


for i in range(1,10):
    reg_gamma = SVC(gamma=i)
    reg_gamma.fit(X_train, y_train)
    a = reg_gamma.score(X_test, y_test)
    print('scores: {}'.format(a))

for i in range(1,10):
    reg_C = SVC(C=i)
    reg_C.fit(X_train, y_train)
    a = reg_C.score(X_test, y_test)
    print('scores: {}'.format(a))



