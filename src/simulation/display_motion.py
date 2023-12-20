''''This module is used in visualizer class to display animation'''
from numpy import (
    vstack, min, max, array, zeros, mean
)
from matplotlib.pyplot import figure, show
from matplotlib.animation import FuncAnimation

class DisplayMotion:
    '''This class display motion of particles using FuncAnimation'''
    def __init__(self, particle_positions, fps=5):
        
        self.fps = fps
        
        self.particle_positions = particle_positions
        self.fig = figure(figsize=(15, 8))  # Increase the figure size
        self.ax_convergence = self.fig.add_subplot(121)  # Add 2D subplot on the left for convergence curve
        self.line_convergence, = self.ax_convergence.plot([], [], color='red', label='Convergence Curve')
        self.scatter_point, = self.ax_convergence.plot([], [], 'ro', label='Iteration Best Value')
        self.line_best_values, = self.ax_convergence.plot([], [], color='blue', linestyle='dashed', label='Iteration Best Values')
        self.line_mean_cost, = self.ax_convergence.plot([], [], color='green', label='Mean cost of population')
        self.ax_motion = self.fig.add_subplot(122, projection='3d')  # Add 3D subplot on the right
        self.sc = None  # Variable to store scatter plot object
        self.animation = None

        # Determine dynamic xlim, ylim, and zlim
        all_positions = vstack(self.particle_positions)
        self.ax_motion.set_xlim([min(all_positions[:, 0]), max(all_positions[:, 0])])
        self.ax_motion.set_ylim([min(all_positions[:, 1]), max(all_positions[:, 1])])
        self.ax_motion.set_zlim([min(all_positions[:, 2]), max(all_positions[:, 2])])

        # Set isometric view
        self.ax_motion.view_init(elev=30, azim=45)
        
          # Set dynamic xlim for convergence plot
        iters = len(self.particle_positions)
        self.ax_convergence.set_xlim([1, iters])  # Set xlim from 1 to the number of iterations
        self.__set_ylim()  # Dynamically set ylim based on overall min and max of iter_best_values

        self.ax_motion.set_xlabel('X-axis')
        self.ax_motion.set_ylabel('Y-axis')
        self.ax_motion.set_zlabel('Cost')
        self.ax_motion.set_title('Particle Motion and Cost in PSO Optimization')
        self.ax_motion.legend()

        self.ax_convergence.set_xlabel('Iteration')
        self.ax_convergence.set_ylabel('Objective Value')
        self.ax_convergence.set_title('Particle Swarm Optimization')
        self.ax_convergence.legend()
        
        self.__create_animation()

    def __update_frame(self, frame):
        '''Iterate through list of matrixes to scatter position of particles'''
        
        if self.sc is not None:
            self.sc.remove()

        # Plot the current frame in 3D subplot
        particle_positions = vstack(self.particle_positions[frame])
        self.sc = self.ax_motion.scatter(
            particle_positions[:, 0],
            particle_positions[:, 1],
            particle_positions[:, 2],
            c=particle_positions[:, 2],
            cmap='viridis',
            label='Particles'
        )

        # Update convergence curve and iteration best values in 2D subplot
        iter_best_values, convergence_curve, mean_cost = self.__get_convergence_curve(frame)
        self.line_convergence.set_data(range(frame + 1), convergence_curve)
        self.scatter_point.set_data(frame + 1, iter_best_values[frame])  # +1 to align with 1-based iteration
        self.line_best_values.set_data(range(frame + 1), iter_best_values)
        self.line_mean_cost.set_data(range(frame + 1), mean_cost)

    def __create_animation(self):
        '''Initialize animation of motion'''
        self.animation = FuncAnimation(
            self.fig,
            self.__update_frame,
            frames=len(self.particle_positions),
            interval=1000 / self.fps,  
            repeat=False
        )
        show()
        
    def __get_convergence_curve(self, current_frame):
        iters = current_frame + 1
        if iters == 0:
            return array([]), array([])

        iter_best_values = zeros(iters)
        mean_cost = zeros(iters)

        for i, matrix in enumerate(self.particle_positions[:current_frame + 1]):
            iter_best_values[i] = min(matrix[:, 2])
            mean_cost[i] = mean(matrix[:, 2])

        convergence_curve = iter_best_values.copy()

        for i in range(1, convergence_curve.shape[0]):
            if convergence_curve[i] > convergence_curve[i - 1]:
                convergence_curve[i] = convergence_curve[i - 1]

        return iter_best_values, convergence_curve, mean_cost

    def __set_ylim(self):
        # Compute overall min and max of iter_best_values
        all_mean_cost = array([mean(matrix[:, 2]) for matrix in self.particle_positions])
        self.ax_convergence.set_ylim([0, max(all_mean_cost) * 1.1])
