import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx  # For braid/cycle detection

# Parameters (small scale for prototype; scale up for larger sims)
NUM_LAYERS = 3  # 3 layers for 3D HCP
POINTS_PER_LAYER = 9  # 3x3 base for small grid
L_P = 1.0  # Planck length unit
QI_THETA = 8.8e26  # QI horizon
C = 3e8  # Speed of light unit
MU0 = 4 * np.pi * 1e-7  # Vacuum permeability
DT = 1e-3  # Time step (arbitrary units)
STEPS = 5000  # Reduced steps for prototype

# Generate 3D HCP lattice points (A-B-A stacking)
def generate_hcp_points(num_layers, points_per_layer_side):
    points = []
    for layer in range(num_layers):
        offset_x = (layer % 2) * 0.5 * L_P  # Shift for HCP
        offset_y = (layer % 2) * (np.sqrt(3)/6) * L_P
        for i in range(points_per_layer_side):
            for j in range(points_per_layer_side):
                x = (i + 0.5 * j) * L_P + offset_x
                y = j * (np.sqrt(3)/2) * L_P + offset_y
                z = layer * (np.sqrt(6)/3) * L_P  # Layer spacing for close-packing
                points.append([x, y, z])
    return np.array(points)

points = generate_hcp_points(NUM_LAYERS, int(np.sqrt(POINTS_PER_LAYER)))

# Build graph: Edges between neighbors (<1.1 L_P for hex connectivity)
dist_matrix = cdist(points, points)
edges = np.argwhere((dist_matrix < 1.1 * L_P) & (dist_matrix > 0))
num_edges = len(edges)

# Initialize currents (I) and twists (topological labels ±1)
currents = np.random.uniform(-1, 1, num_edges)  # Scalar currents
twists = np.random.choice([-1, 1], num_edges)
velocities = np.zeros(num_edges)  # For Verlet (dI/dt)

# Positions for Verlet (treat currents as "positions" for analogy)
positions = currents.copy()  # Rename for clarity

# Simplified Ampere-Weber force (scalar mag, longitudinal/transverse approx)
def compute_forces(pos, points, edges):
    forces = np.zeros(len(edges))
    for idx, (i, j) in enumerate(edges):
        r_vec = points[j] - points[i]
        r = np.linalg.norm(r_vec)
        if r == 0: continue
        # Simplified retarded time (assume small dt, approx)
        ret_time = r / C
        # Mock I1, I2 at retarded (use current pos for proto)
        I1 = pos[idx]  # Current on this edge
        I2_avg = np.mean([pos[k] for k in range(len(edges)) if k != idx])  # Avg neighbor
        # Ampere-Weber approx: F ~ mu0 I1 I2 / (4 pi r^2) * (trans/long terms)
        cos_theta = np.random.uniform(-1, 1)  # Mock angles
        long_term = 2 * np.cos(np.pi/2) - 3 * cos_theta**2  # Longitudinal
        trans_term = 1 - (ret_time * DT)**2 / (2 * C**2)  # Retard approx
        forces[idx] = -MU0 * I1 * I2_avg / (4 * np.pi * r**2) * long_term * trans_term
    return forces

# QI-modified mass
def qi_mass(acc, m0=1.0):
    abs_acc = np.abs(acc)
    abs_acc[abs_acc == 0] = 1e-10  # Avoid div0
    return m0 * (1 - 2 * C**2 / (QI_THETA * abs_acc))

# Velocity Verlet integration
for step in range(STEPS):
    forces = compute_forces(positions, points, edges)
    acc = forces / qi_mass(velocities / DT)  # Effective acc with QI
    velocities += 0.5 * acc * DT
    positions += velocities * DT
    forces_new = compute_forces(positions, points, edges)
    acc_new = forces_new / qi_mass(velocities / DT)
    velocities += 0.5 * acc_new * DT

# Braid detection: Build graph, find cycles with twist sum !=0
G = nx.Graph()
G.add_edges_from(edges)
cycles = nx.cycle_basis(G)
braid_edges = []
for cycle in cycles:
    cycle_edges = [(cycle[i], cycle[(i+1)%len(cycle)]) for i in range(len(cycle))]
    twist_sum = sum(twists[np.where((edges == e).all(1) | (edges == e[::-1]).all(1))[0][0]] for e in cycle_edges)
    if twist_sum != 0:
        braid_edges.append(len(cycle_edges))

print("Detected braid edge counts:", braid_edges)

# Mock mass mapping (as in Paper 2)
masses = [1.0 if e<7 else 200 if e<13 else 'variable' for e in braid_edges]
print("Mock particle types/masses:", masses)
