import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 

#chargement des données
data = pd.read_csv("dataset.csv",header=0)
print(data.shape)

#Adaptation des données :
feature_cols = data.columns
feature_cols = feature_cols[0:len(feature_cols)-1]
X = data[feature_cols]
y = data['class']
Scaller = preprocessing.StandardScaler().fit(X)
X = Scaller.transform(X)

# Division des données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#  ---------------- Decision Tree : ----------------
# Création de l'arbre de décision :
deci_tree = DecisionTreeClassifier(criterion = 'gini',splitter="best",max_depth=10,min_samples_split=15, min_samples_leaf=5)
# Entrainement de l'arbre :
deci_tree = deci_tree.fit(X_train,y_train)
# Tester l'arbre :
y_pred = deci_tree.predict(X_test)
# Evaluation de l'arbre :
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Visualisation de l'arbre :
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(deci_tree,filled=True)
# ----------------------------------------------------

# ----------------------- ANN -----------------------
# Création de réseau de neurons :
print('\nRéseau de neurons :\n')
ann = MLPClassifier(solver='lbfgs', alpha=1, hidden_layer_sizes=(4),
                    activation='logistic', max_iter=1000,random_state=1 )
# Entrainement du modéle :
ann.fit(X_train, y_train.values.ravel())
# Tester le modéle :
predictions = ann.predict(X_test)

# Evaluation du modéle :
print('Matrice de confusion : ')
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
# ----------------------------------------------------

# ----------------------- SVM Linear -----------------------
# Construction du modéle
print('\nSVM linéaire :\n')
svc_lin = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', coef0=5.0,
                   shrinking=False, probability=True, tol=0.001, cache_size=200, 
                   class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovo', 
                   break_ties=False, random_state=None)
svc_lin.fit(X_train, y_train)
y_pred = svc_lin.predict(X_test)
score= svc_lin.score(X_train,y_train)
# Evaluation de l'SVM :
print('Matrice de confusion : ')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# La validation croisé
model = cross_val_score(svc_lin, X, y, cv=5)
print(model)
# on a choisit k=3
print(model[2])
# ----------------------------------------------------

# ----------------------- SVm Poly -----------------------
# Construction du modéle
print('\nSVM polynome :\n')
svc_poly = SVC(C=1.0, kernel='poly', degree=3, gamma='scale', coef0=5.0,
                   shrinking=False, probability=True, tol=0.001, cache_size=200, 
                   class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovo', 
                   break_ties=False, random_state=None)
svc_poly.fit(X_train, y_train)
y_pred = svc_poly.predict(X_test)
score= svc_poly.score(X_train,y_train)

# Evaluation de l'SVM :
print('Matrice de confusion : ')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# La validation croisé
model = cross_val_score(svc_poly, X, y, cv=5)
print(model)

# on a choisit k=5
print(model[4])
# ----------------------------------------------------

# ----------------------- SVm Rbf -----------------------
# Construction du modéle
print('\nSVM Rbf :\n')
svclassifier = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=5.0,
                   shrinking=False, probability=True, tol=0.001, cache_size=200, 
                   class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovo', 
                   break_ties=False, random_state=None)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
score= svclassifier.score(X_train,y_train)
# Evaluation de l'SVM :
print('Matrice de confusion : ')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# La validation croisé
model = cross_val_score(svclassifier, X, y, cv=5)
print(model)
# on a choisit k=5
print(model[4])
# ----------------------------------------------------
