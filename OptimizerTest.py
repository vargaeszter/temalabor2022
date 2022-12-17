# %%
# beolvasas
import pandas as pd
import numpy as np

dataset = pd.read_csv("/workspaces/python_starter/heart_disease_health_indicators.csv")

# %%
# betanito es teszt adathalmazok szétválasztása

from sklearn.model_selection import train_test_split 

X = dataset.iloc[:,1:21]
y = np.ravel(dataset.HeartDiseaseorAttack)     # Specify the target labels and flatten the array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size = 0.1, random_state=42)

# standardizalas

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
# modellkeszites - for SGD

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(21, activation='relu', input_shape=(20,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Model output shape
model.output_shape

# Model summary
model.summary()

# %%
# forditas és illesztes - SGD
import keras

f = open("OptTestSGD.txt", "w")
LearningRate = [0.001, 0.0001, 0.00001]
for x in LearningRate:
    model.compile(loss='binary_crossentropy',
        optimizer=keras.optimizers.SGD(learning_rate=x),
        metrics=['accuracy']
    )
    model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)
    print("SGD: ", x, "\t")
    content = str(x)
    f.write(content)
    f.write("\t")
    score = model.evaluate(X_test, y_test,verbose=1)
    print(score, "\n")
    content = str(score)
    f.write(content)
    f.write("\n")

f.close()

# %%
# modellkeszites - for RMSprop
model = Sequential()

model.add(Dense(21, activation='relu', input_shape=(20,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# %%
# forditas és illesztes - RMSprop

f = open("OptTestRMSprop.txt", "w")

for x in LearningRate:
    model.compile(loss='binary_crossentropy',
        optimizer=keras.optimizers.RMSprop(learning_rate=x),
        metrics=['accuracy']
    )
    model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)
    print("RMSpropt: ", x, "\t")
    content = str(x)
    f.write(content)
    f.write("\t")
    score = model.evaluate(X_test, y_test,verbose=1)
    print(score, "\n")
    content = str(score)
    f.write(content)
    f.write("\n")

f.close()

# %%
# modellkeszites - for Adam
model = Sequential()

model.add(Dense(21, activation='relu', input_shape=(20,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# %%
# forditas és illesztes - Adam

f = open("OptTestAdam.txt", "w")

for x in LearningRate:
    model.compile(loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=x),
        metrics=['accuracy']
    )
    model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)
    print("Adam: ", x, "\n")
    content = str(x)
    f.write(content)
    f.write("\t")
    score = model.evaluate(X_test, y_test,verbose=1)
    print(score, "\n")
    content = str(score)
    f.write(content)
    f.write("\n")

f.close()