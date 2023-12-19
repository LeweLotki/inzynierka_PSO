import logging
import numpy as np
import multiprocessing as mp
from collections import deque
from pyswarms.backend.operators import compute_pbest, compute_objective_function
from pyswarms.backend.topology import Star
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler, OptionsHandler
from pyswarms.base import SwarmOptimizer
from pyswarms.utils.reporter import Reporter

class GlobalBestPSOWithCallback(SwarmOptimizer):
    
    '''
    This class inherits from SwarmOptimizer to create possibility
    of calling callback function in each iteration of algorithm training
    Also it allows to return position of each particle from the swarm
    '''
    
    def __init__(self, n_particles, dimensions, options, bounds=None, oh_strategy=None, bh_strategy="periodic", velocity_clamp=None, vh_strategy="unmodified", center=1.00, ftol=-np.inf, ftol_iter=1, init_pos=None, callback=None):
        super(GlobalBestPSOWithCallback, self).__init__(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds, velocity_clamp=velocity_clamp, center=center, ftol=ftol, ftol_iter=ftol_iter, init_pos=init_pos)
        
        if oh_strategy is None:
            oh_strategy = {}
        
        self.rep = Reporter(logger=logging.getLogger(__name__))
        self.callback = callback
        
        self.reset()
        
        self.top = Star()
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.oh = OptionsHandler(strategy=oh_strategy)
        self.name = __name__

    def optimize(self, objective_func, iters, n_processes=None, verbose=True, **kwargs):
        # Existing code...
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """
        
        particle_positions = []  # Store particle positions

        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )
        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            # fmt: off
            self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=pool, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            # Set best_cost_yet_found for ftol
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
            # fmt: on
            if verbose:
                self.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            )
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform options update
            self.swarm.options = self.oh(
                self.options, iternow=i, itermax=iters
            )
            
            # Call the callback function
            if self.callback is not None:
                # Calculate cost function value for each particle
                current_cost_column = np.array([objective_func(pos) for pos in self.swarm.position]).reshape(-1, 1)
                # Store positions with cost for each particle
                positions_with_cost = np.column_stack((self.swarm.position, current_cost_column))
                particle_positions.append(positions_with_cost.copy())
                # Call the user-provided callback function
                self.callback(positions_with_cost)
            
            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()
        return (final_best_cost, final_best_pos, particle_positions)

# # Example quadratic objective function
# def quadratic_objective(x):
#     return np.sum(x**2)

# # Example callback function
# def my_callback(particle_positions):
#     # Collect particle positions
#     print(particle_positions.copy())

# # Example usage
# options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
# all_positions = []  # Store particle positions
# optimizer = GlobalBestPSOWithCallback(n_particles=10, dimensions=2, options=options, callback=my_callback)
# best_cost, best_position, particle_positions = optimizer.optimize(quadratic_objective, iters=100)

# # Display optimization results
# print("Optimization Results:")
# print("Best Cost:", best_cost)
# print("Best Position:", best_position)

# # Visualize 3D scatter plot
# particle_positions = np.vstack(particle_positions)
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(particle_positions[:, 0], particle_positions[:, 1], particle_positions[:, 2], c=particle_positions[:, 2], cmap='viridis', label='Particles')
# ax.scatter(best_position[0], best_position[1], best_cost, color='red', marker='x', label='Global Best')

# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Cost')
# ax.set_title('Particle Motion and Cost in PSO Optimization')
# ax.legend()
# plt.show()
