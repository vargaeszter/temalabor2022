# %%
# beolvasas
import pandas as pd
import numpy as np

dataset = pd.read_csv("/workspaces/python_starter/heart_disease_health_indicators.csv")

# %%
# adat szerkezet vizsgálata
# dataset.head(4)   # első N adat
# dataset.tail(5)   # utolsó N adat
dataset.sample(5)   # random N adat

dataset.describe()  # általános statisztikai összesítés

#pd.isnull(dataset)     # NULL elemek kezelése

# %%
#Visualization Occurence of Disease

import matplotlib.pyplot as plt

dataset.groupby(['HeartDiseaseorAttack']).mean()

count = dataset.HeartDiseaseorAttack.value_counts().values
#label = dataset.PhysActivity.value_counts().index
label = ['Healthy', 'HeartDiseaseOrAttack']

colorlist = ['C2', 'C1']

plt.pie(count, labels=label, colors=colorlist, autopct='%1.1f%%',
    startangle=180, shadow=True)

plt.title("Occurence of Heart Disease or Attack")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Sources
# https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html?fbclid=IwAR3Ghaokw2s5G6hMYwKuyFsaE838jEUnGeMHN3wGiApod44PIHDg_sbYD4g
# https://matplotlib.org/

# %%
# vizualizacio#4: korrelacios matrix
import seaborn as sns
corr = dataset.corr()
sns.heatmap(corr, 
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    cmap='coolwarm',
    fmt='.5g',
)

# %%
#modellkeszites a szivebetegsegkockazat elorejelzesere a k folf elotti modszerekkel
# betanito es teszt adathalmazok

# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Specify the data 
X = dataset.iloc[:,1:21]

# Specify the target labels and flatten the array
# https://numpy.org/doc/stable/reference/generated/numpy.ravel.html
y = np.ravel(dataset.HeartDiseaseorAttack)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size = 0.7, random_state=42)

# %%
# standardizalas

# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

# %%
# modellkeszites

# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(21, activation='relu', input_shape=(20,)))

# Add two hidden layer 
model.add(Dense(16, activation='relu'))
model.add(Dense(11, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

# Model output shape
model.output_shape

# Model summary
model.summary()

# Sources
# https://www.datacamp.com/tutorial/deep-learning-python
# https://keras.io/api/layers/core_layers/dense/
# https://keras.io/api/layers/core_layers/input/https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# https://keras.io/api/layers/activations/

# %%
# forditas és illesztes
import keras

model.compile(loss='binary_crossentropy',
 optimizer=keras.optimizers.Adam(learning_rate=1e-3),
 metrics=['accuracy']
)
                   
model.fit(X_train, y_train, epochs=15, batch_size=1, verbose=1)

# %%
#modell kiértékelése

score = model.evaluate(X_test, y_test,verbose=1)
print(score)

# %%
# saving test data to files for later 
# https://www.geeksforgeeks.org/how-to-save-a-numpy-array-to-a-text-file/#:~:text=Let%20us%20see%20how%20to,array%20to%20a%20text%20file.&text=Creating%20a%20text%20file%20using,file%20using%20close()%20function.

f = open("preds.txt", "w")
for x in y_pred:
    content = str(x)
    f.write(content)
    f.write("\n")
f.close()

content = str(y_test)
f = open("test.txt", "w")
for x in y_test:
    content = str(x)
    f.write(content)
    f.write("\n")
f.close() 

# %%
#Threshold select visualization
import matplotlib.pyplot as plt

barlabels = ['correct 0', 'false alarm', 'undetected', 'correct 1']
colorlist = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']

data1 = [39496, 29511, 581, 6516]    # 0.05	    39496	      29511	       581	     6516
data2 = [60958, 8049, 3194, 3903]   # 0.2	    60958	       8049	      3194	     3903
data3 = [68688, 319, 6619, 478]
data4 = [0.99537728, 0.0046227194, 0.9326476, 0.067352402]

fig, ax = plt.subplots()
ax.bar(barlabels,data1,1,color = colorlist, label = barlabels)
ax.set_title("Multiplicity of deciosions, threshold = 0.05")
plt.show()

fig, ax = plt.subplots()
ax.bar(barlabels,data2,1,color = colorlist, label = barlabels)
ax.set_title("Multiplicity of deciosions, threshold = 0.2")
plt.show()

fig, ax = plt.subplots()
ax.bar(barlabels,data3,1,color = colorlist, label = barlabels)
ax.set_title("Multiplicity of deciosions, threshold = 0.5")
plt.show()

fig, ax = plt.subplots()
ax.bar(barlabels,data4,1,color = colorlist, label = barlabels)
ax.set_title("Probability of deciosions, threshold = 0.5")
plt.show()


