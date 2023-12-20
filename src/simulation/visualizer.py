'''Definition of Visualizer class'''
from numpy import (
    array, vstack, min, zeros
)
from matplotlib.pyplot import (
    figure, plot, scatter, xlabel,
    ylabel, title, legend, show
)

from simulation.display_motion import DisplayMotion

class Visualizer:
    '''This class is created to visualize motion of particles in different ways'''
    def __init__(self, particle_positions):
        
        self.particle_positions = particle_positions
    
    def __get_convergence_curve(self):
        
        iters = len(self.particle_positions) 
        
        if iters == 0:
            return array([])
        
        iter_best_values = zeros(iters)
        
        for i, matrix in enumerate(self.particle_positions):
            iter_best_values[i] = min(matrix[:, 2])

        convergence_curve = iter_best_values.copy()
        
        for i in range(1, convergence_curve.shape[0]):
            if convergence_curve[i] > convergence_curve[i - 1]:
                convergence_curve[i] = convergence_curve[i - 1]
        
        return (iter_best_values, convergence_curve)
        
    def display_convergence(self):
        '''display convergence curve'''

        iter_best_values, convergence_curve = self.__get_convergence_curve()

        figure(figsize=(8, 6))
        plot(convergence_curve, color='red')
        scatter(range(convergence_curve.shape[0]), convergence_curve, c='r', label='convergence curve')
        plot(iter_best_values, color='blue', linestyle='dashed')
        scatter(range(iter_best_values.shape[0]), iter_best_values, c='b', label='iteration best value')
        xlabel('Iteration')
        ylabel('Objective Value')
        title('Particle Swarm Optimization')
        legend()
        show()
        
    def display_particle_positions(self):
        '''display position of each particle on 3D scatter plot'''
        
        self.particle_positions = vstack(self.particle_positions)
        fig = figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.particle_positions[:, 0], self.particle_positions[:, 1], self.particle_positions[:, 2], c=self.particle_positions[:, 2], cmap='viridis', label='Particles')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Cost')
        ax.set_title('Particle Motion and Cost in PSO Optimization')
        ax.legend()
        show()
        
    def display_motion(self):
        '''association of DisplayMotion class'''
        DisplayMotion(self.particle_positions)