from ast import literal_eval
from numpy import (
    array, ndarray
)
from pandas import (
    DataFrame, read_csv
)

from scipy.spatial import cKDTree

from simulation.GlobalBestPSOWithCallback import GlobalBestPSOWithCallback
from simulation.visualizer import Visualizer
from config import paths

class PSO:
    '''train PSO algorithm on chosen cost function and display results'''
    options = {
        'c1': 0.0, 
        'c2': 0.0, 
        'w' : 0.0
        }
    bounds = (
        array([0, 0]), 
        array([0, 0])
        )
    optimzier = None
    objective_values = None
    particle_positions = None
    dimensions = 2
    df = DataFrame()
    tree = None 
    
    def __init__(
        self, 
        file_path:str='cost_function.csv', 
        options:str='(0.09, 0.01, 0.09)', 
        n_particles:int=100, 
        iters:int=15
        ):
        
        try:
            options = literal_eval(options)
        except Exception as e:
            print(e)
        
        self.file_path = paths.cost_functions_folder_path + '/' + file_path
        self.options['c1'] = options[0]
        self.options['c2'] = options[1]
        self.options['w'] = options[2]
        self.n_particles = n_particles
        self.iters = iters
    
        self.__get_dataframe()
        self.__get_optimizer()
        
    def __get_dataframe(self):
        
        file_path = self.file_path
        self.df = read_csv(file_path)
        x_column = self.df['x'].values
        y_column = self.df['y'].values

        self.bounds[0][0] = x_column.min()
        self.bounds[1][0] = x_column.max()
        self.bounds[0][1] = y_column.min()
        self.bounds[1][1] = y_column.max()
        
        self.tree = cKDTree(self.df[['x', 'y']].values)
        
    def __objective_function(self, positions) -> ndarray:
    
        _, indices = self.tree.query(positions, k=1)

        if type(indices) == ndarray: 
            indices.flatten()

        closest_cost_values = self.df['cost'].values[indices]

        return closest_cost_values

    def __get_optimizer(self) -> None: 
    
        self.optimizer = GlobalBestPSOWithCallback(
            n_particles=self.n_particles, 
            dimensions=self.dimensions, 
            options=self.options, 
            bounds=self.bounds,
            callback=lambda particle_positions:None
            )

    def train(self) -> None:
        '''train PSO algorithm'''
        try:
            _, _, self.particle_positions = self.optimizer.optimize(self.__objective_function, iters=self.iters)
        except ValueError as e:
            print("Optimization failed. Make sure the initial swarm has valid positions.")
            print(e)

    def display(self, fps=5) -> None:
        '''association of Visualizer class'''
        visualizer = Visualizer(self.particle_positions)
        # visualizer.display_convergence()
        # visualizer.display_particle_positions()
        visualizer.display_motion(fps=fps)
        
