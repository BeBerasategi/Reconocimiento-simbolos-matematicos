# python3 RF_grid_search.py > log_gs.txt 2> err_gs.txt & 
# DURACIÓN COMPLETA: 

# Se cambia la profundidad del árbol para ver el efecto de la regularización.
# Más tarde, en un segundo intento, se cambia el número de estimadores.

import sys, os, time, functools
sys.path.append('/home/jovyan/work/databases/HASY/')
os.makedirs("./models_gs/", exist_ok=True)

from hasy_tools_updated import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier 

# Save the model:
from joblib import dump, load

start_time = time.time()
print = functools.partial(print, flush=True)


np.random.seed(42)
print('DESCRIPTION: Random Forest training, with grid search for depth.')
print('Start time:', time.strftime("%H:%M:%S", time.gmtime(start_time)), end='\n\n')

X_train, y_train, symbol_id2index_train, index2symbol_id_train = load_database('/home/jovyan/work/databases/HASY/benat-data/cv/fold-1/train.csv')
X_test, y_test, symbol_id2index_test, index2symbol_id_test = load_database('/home/jovyan/work/databases/HASY/benat-data/cv/fold-1/test.csv')
print("Data loaded")

y_train_id = np.array([index2symbol_id_train[element] for element in y_train])
y_test_id = np.array([index2symbol_id_test[element] for element in y_test])
print("Converted y indexes to symbol IDs")

# Data scaling is not done (not needed for Random Forest)
X_train = np.array(X_train) #/255.0
y_train = y_train_id.astype(int)
X_test = np.array(X_test) #/255.0
y_test = y_test_id.astype(int)
# print("Data scaled", end='\n\n')

print("Grid search started-----------------")

acc = []
acc_tr = []
depth_list = [i for i in range(5,56,5)]
for depth in depth_list:
    print(f"depth={depth}")
    rf=RandomForestClassifier(n_estimators=100, max_depth=depth)
    rf.fit(X_train,y_train_id)
    
    # Attemp to save model
    dump(rf, f'./models_gs/RF_HASY_depth_{depth}.joblib') 
    
    p_tr = rf.predict(X_train)
    a_tr = metrics.accuracy_score(y_train_id, p_tr)
    
    pred = rf.predict(X_test)
    a = metrics.accuracy_score(y_test_id, pred)
    
    acc_tr.append(a_tr)
    acc.append(a)

print("Grid search finished-----------------", end='\n\n')

end_time = time.time()
elapsed_time = end_time - start_time
print('End time:', time.strftime("%H:%M:%S", time.gmtime(end_time)), end='\n\n')
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

plt.subplots(figsize=(10, 5))
plt.plot(depth_list, acc,'-D' ,color='red', label="Testing Accuracy")
plt.plot(depth_list, acc_tr,'-gD', label="Training Accuracy")
#plt.xticks(L,L)
plt.grid(True)
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title('Accuracy versus the Max depth')
plt.savefig("Accuracy_vs_max_depth.png", dpi=200)

print("Plot made and saved")

print("Saving data...")

# Assuming depth_list, acc_tr, and acc are lists of the same length
data = {
    "depth": depth_list,
    "acc_tr": [ '%.4f' % elem for elem in acc_tr],
    "acc": [ '%.4f' % elem for elem in acc]
}

df = pd.DataFrame(data)

# To save the dataframe to a csv file
df.to_csv('data_file_depth.csv', index=False)

print("Data saved", end="\n\n")

best_depth = depth_list[np.argmax(acc)]
print("Best value for depth: ", best_depth)
print(f"Best model: ./models_gs/RF_HASY_depth_{depth}.joblib", end='\n\n')

print("----------------------------------------------------------------------------------", end='\n\n')

print('DESCRIPTION: Random Forest training, with grid search for number of estimators.')
print('Best depth from previous grid search is used.')

start_time = time.time()
print('Start time:', time.strftime("%H:%M:%S", time.gmtime(start_time)), end='\n\n')

print("Grid search started-----------------")

acc = []
acc_tr = []
estimator_list = [i for i in range(10,151,10)]
for estimator in estimator_list:
    print(f"estimator={estimator}")
    rf=RandomForestClassifier(n_estimators=estimator, max_depth=best_depth)
    rf.fit(X_train,y_train_id)
    
    # Attemp to save model
    dump(rf, f'./models_gs/RF_HASY_estimators_{estimator}.joblib') 
    
    p_tr = rf.predict(X_train)
    a_tr = metrics.accuracy_score(y_train_id, p_tr)
    
    pred = rf.predict(X_test)
    a = metrics.accuracy_score(y_test_id, pred)
    
    acc_tr.append(a_tr)
    acc.append(a)

print("Grid search finished-----------------", end='\n\n')

end_time = time.time()
elapsed_time = end_time - start_time
print('End time:', time.strftime("%H:%M:%S", time.gmtime(end_time)), end='\n\n')
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

plt.subplots(figsize=(10, 5))
plt.plot(estimator_list, acc,'-D' ,color='red', label="Testing Accuracy")
plt.plot(estimator_list, acc_tr,'-gD', label="Training Accuracy")
#plt.xticks(L,L)
plt.grid(True)
plt.xlabel("Number of estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.title('Accuracy versus the number of estimators')
plt.savefig("Accuracy_vs_n_estimators.png", dpi=200)

print("Plot made and saved")

print("Saving data...")

# Assuming estimator_list, acc_tr, and acc are lists of the same length
data = {
    "n_estimators": estimator_list,
    "acc_tr": [ '%.4f' % elem for elem in acc_tr],
    "acc": [ '%.4f' % elem for elem in acc]
}

df = pd.DataFrame(data)

# To save the dataframe to a csv file
df.to_csv('data_file_estimators.csv', index=False)

print("Data saved", end="\n\n")

best_estimator = estimator_list[np.argmax(acc)]
print("Best value for n_estimators: ", best_estimator)
print(f"Best model: ./models_gs/RF_HASY_estimators_{best_estimator}.joblib", end='\n\n')