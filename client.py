import numpy as np
import grpc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import time
import generated.sph_optimized_pb2 as sph_optimized_pb2
import generated.sph_optimized_pb2_grpc as sph_optimized_pb2_grpc

class SPHCoordinator:
    def __init__(self, num_particles_per_domain=1000, time_step=0.005):
        self.num_particles_per_domain = num_particles_per_domain
        self.time_step = time_step
        self.boundary_position = 0.0
        self.channels = [grpc.insecure_channel('localhost:50051'),
                         grpc.insecure_channel('localhost2:50051')] # хост второго сревера 
        
        self.stubs = [sph_optimized_pb2_grpc.OptimizedSPHSimulationStub(self.channels[0]),
                     sph_optimized_pb2_grpc.OptimizedSPHSimulationStub(self.channels[1]) ]
        self.domain_data = [None]

    def simulate_step(self):
        responses = []
        for domain_id in range(2):
            request = sph_optimized_pb2.SimulationRequest(
                time_step=self.time_step,
                boundary_position=self.boundary_position
            )
            response = self.stubs[domain_id].SimulateStep(request)
            responses.append(response)
        
        #Обмен граничными частицами между доменами
        self._exchange_boundary_particles(responses)
        
        self.domain_data = responses
        

    def _exchange_boundary_particles(self, responses):
        boundary_threshold = 2.0
        for source_domain in range(2):
            target_domain = 1 - source_domain
            boundary_particles_indices = []
            positions_x = responses[source_domain].positions_x
            for i, x in enumerate(positions_x):
                if (source_domain == 0 and x > self.boundary_position - boundary_threshold) or \
                   (source_domain == 1 and x < self.boundary_position + boundary_threshold):
                    boundary_particles_indices.append(i)
            if boundary_particles_indices:
                request = sph_optimized_pb2.BoundaryRequest(
                    positions_x=[positions_x[i] for i in boundary_particles_indices],
                    positions_y=[responses[source_domain].positions_y[i] for i in boundary_particles_indices],
                    positions_z=[responses[source_domain].positions_z[i] for i in boundary_particles_indices],
                    velocities_x=[responses[source_domain].velocities_x[i] for i in boundary_particles_indices],
                    velocities_y=[responses[source_domain].velocities_y[i] for i in boundary_particles_indices],
                    velocities_z=[responses[source_domain].velocities_z[i] for i in boundary_particles_indices],
                    masses=[1.0] * len(boundary_particles_indices),
                    densities=[responses[source_domain].densities[i] for i in boundary_particles_indices],
                    pressures=[responses[source_domain].pressures[i] for i in boundary_particles_indices],
                    source_domain=source_domain
                )
                self.stubs[target_domain].ExchangeBoundaryParticles(request)

    def _visualize_particles(self):
        if self.domain_data[0] is None:
            return np.zeros((600, 800, 3), dtype=np.uint8)
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        for domain_id in range(2):
            response = self.domain_data[domain_id]
            
            positions_x = response.positions_x
            positions_y = response.positions_y
            positions_z = response.positions_z
            
            densities = response.densities
            pressures = response.pressures

 
            tri=ax.scatter(
                positions_x, positions_y, positions_z,
                c=pressures
            )
            fig.colorbar(tri, ax=ax, label='pressures')
        
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('SPH Simulation')
        
        plt.tight_layout()

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, [1, 2, 3]]
        plt.close(fig)

        return image
    
    def run_simulation(self, num_steps=100, output_file='ts_simulation.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        frame_size = (800, 600)
        out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

        for step in range(num_steps):
            self.simulate_step()
            frame = self._visualize_particles()
            out.write(frame)

        out.release()
        print(f"Video saved to {output_file}")
