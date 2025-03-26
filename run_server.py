import sys
from server import serve

if __name__ == "__main__":
    serve(gpu_id=0, domain_id=1, num_particles=5000)