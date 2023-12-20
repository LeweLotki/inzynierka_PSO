''''This module is used in visualizer class to display animation'''
from numpy import (
    vstack, min, max
)
from matplotlib.pyplot import figure, show
from matplotlib.animation import FuncAnimation

class DisplayMotion:
    '''This class display motion of particles using FuncAnimation'''
    def __init__(self, particle_positions, fps=5):
        self.particle_positions = particle_positions
        self.fig = figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.sc = None  
        self.animation = None
        self.fps = fps
        
        all_positions = vstack(self.particle_positions)
        self.ax.set_xlim([min(all_positions[:, 0]), max(all_positions[:, 0])])
        self.ax.set_ylim([min(all_positions[:, 1]), max(all_positions[:, 1])])
        self.ax.set_zlim([0, max(all_positions[:, 2])])

        self.ax.view_init(elev=30, azim=45)  
        
        self.__create_animation()

    def __update_frame(self, frame):
        '''Iterate through list of matrixes to scatter position of particles'''
        
        if self.sc is not None:
            self.sc.remove()

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
            interval=1000 / self.fps,  
            repeat=False
        )
        show()
