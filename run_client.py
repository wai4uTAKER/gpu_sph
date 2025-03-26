from client import SPHCoordinator

if __name__ == "__main__":
    coordinator = SPHCoordinator(num_particles_per_domain=200, time_step=0.1)
    coordinator.run_simulation(num_steps=200, output_file='simulation.mp4')


