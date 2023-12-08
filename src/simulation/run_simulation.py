import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarms.single import GlobalBestPSO
from scipy.interpolate import griddata


def objective_function(positions):
    
    x_values = positions[:, 0]
    y_values = positions[:, 1]

    
    file_path = '../data_postprocessing/density_values.csv'
    df = pd.read_csv(file_path)
    x_column = df['x'].values
    y_column = df['y'].values
    density_column = df['density'].values

    
    objective_values = - griddata((x_column, y_column), density_column, (x_values, y_values), method='linear')

    return objective_values  


options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'minimize': False}


bounds = (np.array([1, 0]), np.array([4500, 3500]))


optimizer = GlobalBestPSO(n_particles=100, dimensions=2, options=options, bounds=bounds)


try:
    best_position = optimizer.optimize(objective_function, iters=15)
except ValueError as e:
    print("Optimization failed. Make sure the initial swarm has valid positions.")
    print(e)
    best_position = np.zeros(2)  


history = np.array(optimizer.cost_history)

plt.figure(figsize=(8, 6))
plt.plot(history, c='r')
plt.scatter(range(len(history)), history, c='r', label='Best value')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Particle Swarm Optimization')
plt.legend()
plt.show()
