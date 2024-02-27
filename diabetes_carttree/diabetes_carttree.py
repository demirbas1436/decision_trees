import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
#import graphviz
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("diabetes_cart.csv")
df = pd.DataFrame(df[:20])
print(df)
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)
y_pred = cart_model.predict(X)
y_prob = cart_model.predict_proba(X)[:, 1]
###############################################################################
# Aslında şu soruyu sormak istiyorum, Ağacı bölmeye nereden başlamalıyız?
# Bunun için information gain maksimum olan feature seçilmelidir.
# information gain’i maksimum yapan feature için de Entropy hesaplanmalıdır.
###############################################################################

cart_params = {"max_depth": range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)
cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
#cart_best_grid.best_params_

#cart_best_grid.best_score_
cart_model2 = DecisionTreeClassifier(random_state=1, max_depth=4, min_samples_split=10, criterion="entropy").fit(X, y)


feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
class_names = ["0", "1"]

from sklearn import tree
plt.figure(figsize=(20, 10), dpi=100)
tree.plot_tree(cart_model2, feature_names=feature_names, rounded=True, filled=True, class_names=True, impurity=True)
#plt.show()
#plt.savefig("cart_tree")


print(df[df["DiabetesPedigreeFunction"] <= 0.216])