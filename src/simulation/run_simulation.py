from ast import literal_eval
from numpy import (
    array, ndarray
)
from pandas import (
    DataFrame, read_csv
)
from matplotlib.pyplot import (
    figure, plot, scatter, xlabel,
    ylabel, title, legend, show
)
from pyswarms.single import GlobalBestPSO
from scipy.interpolate import griddata
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
    dimensions = 2
    df = DataFrame()
    
    def __init__(
        self, 
        file_path:str='cost_function.csv', 
        options:str='(0.5, 0.3, 0.9)', 
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
        self.train()
        self.display_convergence()
        
    def __get_dataframe(self):
        
        file_path = self.file_path
        self.df = read_csv(file_path)
        x_column = self.df['x'].values
        y_column = self.df['y'].values

        self.bounds[0][0] = x_column.min()
        self.bounds[1][0] = x_column.max()
        self.bounds[0][1] = y_column.min()
        self.bounds[1][1] = y_column.max()
        
    def __objective_function(self, positions) -> ndarray:
        
        x_values = positions[:, 0]
        y_values = positions[:, 1]

        x_column = self.df['x'].values
        y_column = self.df['y'].values
        cost_column = self.df['cost'].values
        
        objective_values = griddata(
            (x_column, y_column),
            cost_column,
            (x_values, y_values),
            method='linear'
            )
        
        return objective_values

    def __get_optimizer(self) -> None: 
    
        self.optimizer = GlobalBestPSO(
            n_particles=self.n_particles, 
            dimensions=self.dimensions, 
            options=self.options, 
            bounds=self.bounds
            )

    def train(self) -> None:
        '''train PSO algorithm'''
        try:
            self.optimizer.optimize(self.__objective_function, iters=self.iters)
        except ValueError as e:
            print("Optimization failed. Make sure the initial swarm has valid positions.")
            print(e)

    def display_convergence(self) -> None:
        '''display convergence curve'''
        history = array(self.optimizer.cost_history)

        figure(figsize=(8, 6))
        plot(history, c='r')
        scatter(range(len(history)), history, c='r', label='Best value')
        xlabel('Iteration')
        ylabel('Objective Value')
        title('Particle Swarm Optimization')
        legend()
        show()
