# python3 CNN_MNIST_aug_1000_val.py > log_aug.txt 2> err_aug.txt & 
# DURACIÓN COMPLETA: 15h 30min.

# Se utiliza el modelo más apropiado elegido tras la cv, para entrenar con la base de datos aumentada. 1000. Se reserva un set de validación para adaptar mejor la cantidad de epochs.

# Modelo: MNIST del libro "Hands on Machine Learning" (Géron).

import sys, os, time, functools
sys.path.append('/home/jovyan/work/databases/HASY/')
os.makedirs("./models_aug/", exist_ok=True)
os.makedirs("./plots_aug/", exist_ok=True)

from hasy_tools_updated import *

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import sklearn.metrics as metrics

start_time = time.time()
print = functools.partial(print, flush=True)

print('DESCRIPTION: CNN training, with data augmentation. Validation is used. Augmented dataset (600). Model: MNIST from the book "Hands on Machine Learning" (Géron).')
print('Start time:', time.strftime("%H:%M:%S", time.gmtime(start_time)), end='\n\n')

#X_train, y_train, symbol_id2index_train, index2symbol_id_train = load_database(f'/home/jovyan/work/databases/HASY/benat-data/train.csv')

X_test, y_test, symbol_id2index_test, index2symbol_id_test = load_database(f'/home/jovyan/work/databases/HASY/benat-data/test.csv')
print(f"Original data loaded")

# Load augmented data (~1min needed). It contains training data + augmented data.
X_train= np.loadtxt("/home/jovyan/work/databases/Data_Augmentation/DA_1000/X_aug.csv", delimiter=",")
y_train_id= np.loadtxt("/home/jovyan/work/databases/Data_Augmentation/DA_1000/y_aug_id.csv", delimiter=",", dtype=str)
print(f"Augmented data loaded")

# First, get the ID of each label. Augmented data already uses this labeling system:
y_test_id = np.array([index2symbol_id_test[element] for element in y_test])
print("Converted y indexes to symbol IDs")

# Merge original and augmented data: -- NOT NEEDED!
# X_train = np.concatenate((X_train, X_aug), axis=0)
# y_train_id = np.concatenate((y_train_id, y_aug_id), axis=0)
# print("All training data concatenated. Length: ", len(y_train_id))

# Go back to indexes, but use the SAME indexing SYSTEM for all:
y_train = np.array([symbol_id2index_test[element] for element in y_train_id])
y_test = np.array([symbol_id2index_test[element] for element in y_test_id])    
print("Converted y symbol IDs to same index system")

# Data scaling
X_train, X_test = X_train / 255., X_test/ 255.
y_train, y_test = y_train.astype(int), y_test.astype(int)
print("Data scaled", end='\n\n')

# Reshape the data to 2D:
X_train = X_train.reshape(X_train.shape[0],32,32)[..., np.newaxis]
X_test = X_test.reshape(X_test.shape[0],32,32)[..., np.newaxis]
# Another option: tf.image.resize(images, [32,32])

# Split the training data in train0 and validation0
index_split = int(len(y_train)*70/80)
X_train0 = X_train[:index_split]
X_val0 = X_train[index_split:]
y_train0 = y_train[:index_split]
y_val0 = y_train[index_split:]

# Reset keras session:
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

# Build the keras modeL: default Conv2 strides: (1,1)
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
    keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(369, activation="softmax")
])

# Model summary:
model.build(X_train.shape) 
print(model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])    

# Callbacks
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,  monitor="val_accuracy", mode = "max", restore_best_weights=False) 
checkpoint = tf.keras.callbacks.ModelCheckpoint(f"./models_aug/best_model.h5", 
                monitor="val_accuracy", mode="max", save_weights_only = False, save_best_only=True, verbose=1)

history = model.fit(X_train0, y_train0, epochs=30, callbacks=[early_stopping_cb, checkpoint],
                    verbose=2,
                   validation_data=(X_val0, y_val0)) # Validation data is provided!

print("The model has been trained.")

#Save history
df = pd.DataFrame(history.history)
df.to_csv(f'./models_aug/history.csv')

# Plot
df.plot(figsize=(8, 5))
plt.grid(True)
#plt.gca().set_ylim(0, 1)
plt.savefig(f"./plots_aug/learning_curves_plot.png", dpi=200)

# Restore best weights:
model.load_weights(f"./models_aug/best_model.h5")

# Test set accuracy: note that y_test are the symbol indexes, not the IDs.
y_proba = model.predict(X_test, verbose=2)
y_pred = np.argmax(y_proba, axis=-1)
y_pred_id = np.array([index2symbol_id_test[element] for element in y_pred])

# Save statistics:
test_acc = {1:[],3:[],5:[], 'MER':[]}

test_acc[1].append(metrics.accuracy_score(y_test, y_pred))
test_acc['MER'].append(MER(y_test_id, y_pred_id))
test_acc[3].append(metrics.top_k_accuracy_score(y_test, y_proba, k=3))
test_acc[5].append(metrics.top_k_accuracy_score(y_test, y_proba, k=5))

print("Statistics computed. Program finished-----------------", end='\n\n')

end_time = time.time()
elapsed_time = end_time - start_time
print('End time:', time.strftime("%H:%M:%S", time.gmtime(end_time)), end='\n\n')
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

# Assuming depth_list, acc_tr, and acc are lists of the same length
df = pd.DataFrame(test_acc)

# To save the dataframe to a csv file
df.to_csv('data_file_aug.csv', index=False)
df.describe().to_csv('data_file_aug_stats.csv')

print("Data saved", end="\n\n")

print("Top1 accuracy:", df[1])
print("Top3 accuracy:", df[3])
print("Top5 accuracy:", df[5])
print("MER:", df['MER'])

print("End of the script")
