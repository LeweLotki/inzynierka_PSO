''''This module is used in visualizer class to display animation'''
from numpy import vstack
from matplotlib.pyplot import figure, show
from matplotlib.animation import FuncAnimation

class DisplayMotion:
    '''This class display motion of particles using FuncAnimation'''
    def __init__(self, particle_positions):
        self.particle_positions = particle_positions
        self.fig = figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.sc = None  # Variable to store scatter plot object
        self.animation = None
        
        self.__create_animation()

    def __update_frame(self, frame):
        '''Iterate through list of matrixes to scatter position of particles'''
        # Clear previous frame
        if self.sc is not None:
            self.sc.remove()

        # Plot the current frame
        particle_positions = vstack(self.particle_positions[frame])
        self.sc = self.ax.scatter(
            particle_positions[:, 0],
            particle_positions[:, 1],
            particle_positions[:, 2],
            c=particle_positions[:, 2],
            cmap='viridis',
            label='Particles'
        )

        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_zlabel('Cost')
        self.ax.set_title('Particle Motion and Cost in PSO Optimization')
        self.ax.legend()

    def __create_animation(self):
        '''Initialize animation of motion'''
        self.animation = FuncAnimation(
            self.fig,
            self.__update_frame,
            frames=len(self.particle_positions),
            interval=200,  # Set the interval between frames in milliseconds
            repeat=False
        )
        show()
