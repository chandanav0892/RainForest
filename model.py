from sklearn.datasets import load_iris

import pickle
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.ensemble import RandomForestClassifier
rfclf = RandomForestClassifier()
rfclf.fit(X,y)

pickle.dump(rfclf, open('model.pickle','wb'))
model = pickle.load(open('model.pickle','rb'))