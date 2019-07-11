from copy import deepcopy
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\student\Temp_ET_Data.csv')

print(data.shape)
data.head()

f1 = data['MeanT'].values
f2 = data['EP'].values

X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='blue', s=2)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
k=3
C_x = np.random.randint(15, np.max(X), size=k)
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32) 
print(C)

plt.scatter(f1, f2, c='#050505', s=2)
plt.scatter(C_x, C_y, marker='o', s=200, c='g')

C_old = np.zeros(C.shape)
error = dist(C, C_old, None)
clusters=np.zeros(f1.shape)
while error != 0:
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
    
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')