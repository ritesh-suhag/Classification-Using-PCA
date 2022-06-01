
##############################################################################
# PCA - CODE TEMPLATE
##############################################################################

# IMPORT REQUIRED PACKAGES 

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# IMPORT SAMPLE DATA 

# IMPORT

data_for_model = pd.read_csv("data/sample_data_pca.csv")

# DROP UNECESSARY COLUMNS

data_for_model.drop(["user_id"], axis = 1, inplace = True)

# SHUFFLE DATA 

data_for_model = shuffle(data_for_model, random_state = 42)

# Checking class balance - 
data_for_model["purchased_album"].value_counts(normalize = True)

# DEAL WITH MISSING VALUES 

data_for_model.isna().sum().sum()

# SPLIT INPUT AND OUTPUT VARIABLES

X = data_for_model.drop("purchased_album", axis = 1)
y = data_for_model["purchased_album"]

# SPLIT OUT TRAINING AND TEST SETS 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FEATURE SCALING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Scaling is vital for PCA.
# We use standardization here.
scale_standard = StandardScaler()

X_train = scale_standard.fit_transform(X_train)
X_test = scale_standard.transform(X_test)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ APPLY PCA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# INSTANTIATE & FIT

# Setting n_components to none will create as many components as ther are columns.
# n_components takes the percentage of variance we want explained.
pca = PCA(n_components = None, random_state = 42)

# Fitting to data - 
pca.fit(X_train)

# EXTRACT THE EXPLAINED VARIANCE ACROSS COMPONENTS

explained_variance = pca.explained_variance_ratio_
explained_variance_cummulative = pca.explained_variance_ratio_.cumsum()

# ~~~~~~~~~~~~~~ PLOT THE EXPLAINED VARIANCE ACROSS COMPONENTS ~~~~~~~~~~~~~~~

# CREATE LIST FOR NUMBER OF COMPONENTS

num_vars_list = list(range(1,101))
plt.figure(figsize = (15,10))

# PLOT THE VARIANCE EXPLAINED BY EACH COMPONENT

plt.subplot(2,1,1)
plt.bar(num_vars_list, explained_variance)
plt.title("Variance across PCA")
plt.xlabel("Number of  Components")
plt.ylabel("% Variance")
plt.tight_layout()

# PLOT THE CUMULATIVE VARIANCE

plt.subplot(2,1,2)
plt.plot(num_vars_list, explained_variance_cummulative)
plt.title("Cummulative Variance across PCA")
plt.xlabel("Number of  Components")
plt.ylabel("Cummulative % Variance")
plt.tight_layout()
plt.show()

# ~~~~~~~~~~~~~~~ APPLY PCA WITH SELECTED NUMBER OF COMPONENTS ~~~~~~~~~~~~~~~

pca = PCA(n_components = 0.75, random_state = 42)
X_train = pca.fit_transform(X_train)
X_test = pca. transform(X_test)

# To get the umber of components created - 
pca.n_components_

# ~~~~~~~~~~~~~~~ APPLY PCA WITH SELECTED NUMBER OF COMPONENTS ~~~~~~~~~~~~~~~

clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train, y_train)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ ACCESS MODEL ACCURACY ~~~~~~~~~~~~~~~~~~~~~~~~~~~

y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)
