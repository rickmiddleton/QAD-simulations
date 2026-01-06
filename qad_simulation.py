import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from collections import defaultdict
import time
import json
from datetime import datetime
from scipy.spatial import KDTree
# ==================== QAD Simulation v8.5 - wrap detection ============================
# Wrap_steps logging for boundary crossings - V8.5 adds particle_density_scale for init
# ======================================================================================
# Create results folder if not exists
results_dir = "qad_simulation_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
# ======================================================================================
# PARAMETERS
# ======================================================================================
code_version = 8.5
mode = 'micro'
macro_optimizations = True
random_seed = 41
force_scale = 1 # was 1.0 - 7.5 works well - been using 8.0 - 15 for zoomy video
base_spacing = 50.0 # 50 was default
layers = 10
I_particle = 1.0
mu0 = 0.5 # was 1.0
v_tang_scale = 1 # was 1500
n_pert = 5
t_max = 7e2
dt = 0.01
Theta_base = 1e25
quantum_blend = 0 # was 0.05, tried 0.1
hbar_sim = 1e-15
num_particles = 50
c_lattice = 3000.0
progress_every = 1000
central_I = 0
swirl_strength = 1 # was 75
freeze_threshold = 0.000001
freeze_patience = 60000
beltrami_damp_factor = 0 # was 0.8 -- been using 0.5
downsample_interval = 1
viscosity_factor = 0 # Damp post-update for balanced energy loss
epsilon_factor = 1.5
fractal_exp = 0.855
history_length = 50 # For retardation; adjust as needed for computation
particle_density_scale = 0.000001
# ======================================================================================
if random_seed is not None:
    random.seed(random_seed)
    np.random.seed(random_seed)
base_G = nx.dodecahedral_graph()
epsilon = base_spacing * epsilon_factor
G = nx.cartesian_product(base_G, nx.path_graph(layers))
N = G.number_of_nodes()
neighbors = dict(G.adj)
L = base_spacing * layers ** 0.25
period = np.array([L, L, L, L])
pos = {}
node_list = list(G.nodes())
base_layout = nx.spring_layout(base_G, dim=3, seed=42)
for node in node_list:
    base_node, layer = node
    base_pos_3d = base_layout[base_node] * (L / 2) + (L / 2)
    pos[node] = np.append(base_pos_3d, (layer / max(1, layers - 1)) * L)
pos_values = np.array([pos[node] % period for node in node_list])
tree = KDTree(pos_values, boxsize=period) # Fixed: Periodic boundaries
def periodic_diff(vec):
    return vec - period * np.round(vec / period)
def qi_mass(a_mag):
    return I_particle * (1 - 2 * c_lattice**2 / (Theta_base * max(a_mag, 1e-10)))
def graph_laplacian_real(f):
    lap = np.zeros(N)
    for i, node in enumerate(node_list):
        nb_list = neighbors[node]
        deg = len(nb_list)
        lap[i] = sum(f[node_list.index(nb)] - f[i] for nb in nb_list) / deg if deg > 0 else 0
    return lap
def compute_helicity(B, J):
    H = 0.0
    for edge in G.edges():
        i, j = node_list.index(edge[0]), node_list.index(edge[1])
        A_approx = (B[i] + B[j]) / 2
        B_avg = (B[i] + B[j]) / 2
        H += np.dot(A_approx, B_avg)
    return H / len(G.edges())
def toroidal_moment(pos, J):
    tm = np.zeros(3) # 3D for cross
    for i in range(N):
        r = pos_values[i][:3] # Slice to 3D
        tm += np.cross(r, J[i][:3])
    return tm / N
def compute_guidance(cur_node, psi):
    nb_list = neighbors[cur_node]
    deg = len(nb_list)
    if deg == 0:
        return np.zeros(4)
    S = np.angle(psi)
    grad_S = np.zeros(4)
    for nb in nb_list:
        nb_idx = node_list.index(nb)
        sep_vec = periodic_diff(pos_values[nb_idx] - pos_values[node_list.index(cur_node)])
        sep = np.linalg.norm(sep_vec)
        if sep > 0:
            grad_S += (S[nb_idx] - S[node_list.index(cur_node)]) * sep_vec / sep**2
    return grad_S / deg
def ampere_force(x, v, cur_node, I_nb, B, J, psi, p):
    nb_list = neighbors[cur_node]
    F = np.zeros(4)
    for idx, nb in enumerate(nb_list):
        nb_idx = node_list.index(nb)
        r_vec = periodic_diff(pos_values[nb_idx] - x)
        sep = np.linalg.norm(r_vec)
        if sep < 1e-8:
            continue
        r_eff = max(1e-8, (sep**2 + epsilon**2) ** (fractal_exp / 2))
        ds1 = r_vec / sep
        ds2 = r_vec / sep # Approximation for ds2
        ds2_norm = np.linalg.norm(ds2)
        ds2_unit = ds2 / ds2_norm if ds2_norm > 0 else np.zeros(4)
        cos_eps = np.dot(ds1, ds2_unit)
        cos_theta1 = np.dot(ds1, ds1) # 1 if unit
        cos_theta2 = np.dot(ds2_unit, ds2_unit) # 1
        long_factor = 2 * cos_eps - 3 * cos_theta1 * cos_theta2
        retard_factor = 1 - np.linalg.norm(v)**2 / (2 * c_lattice**2) # Weber term
        f_mag = -mu0 * I_particle * I_nb[idx] / (4 * np.pi * r_eff) * long_factor * retard_factor * force_scale
        F += f_mag * (r_vec / sep)
    # Add contributions from other particles' nodes (after neighbor loop)
    for other_p in range(num_particles):
        if other_p == p: continue # Skip self
        other_node = cur_nodes[other_p]
        if other_node is None: continue
        other_idx = node_list.index(other_node)
        r_vec = periodic_diff(pos_values[other_idx] - x)
        sep = np.linalg.norm(r_vec)
        if sep < 1e-8: continue
        r_eff = max(1e-8, (sep**2 + epsilon**2) ** (fractal_exp / 2))
        ds1 = r_vec / sep
        ds2 = r_vec / sep # Approx
        ds2_norm = np.linalg.norm(ds2)
        ds2_unit = ds2 / ds2_norm if ds2_norm > 0 else np.zeros(4)
        cos_eps = np.dot(ds1, ds2_unit)
        cos_theta1 = np.dot(ds1, ds1)
        cos_theta2 = np.dot(ds2_unit, ds2_unit)
        long_factor = 2 * cos_eps - 3 * cos_theta1 * cos_theta2
        retard_factor = 1 - np.linalg.norm(v)**2 / (2 * c_lattice**2)
        f_mag = -mu0 * I_particle * I_particle / (4 * np.pi * r_eff) * long_factor * retard_factor * force_scale
        F += f_mag * (r_vec / sep)
    # Add Beltrami alignment (slice to 3D for cross)
    cur_idx = node_list.index(cur_node)
    b_3d = B[cur_idx][:3]
    b_norm = np.linalg.norm(b_3d)
    if b_norm > 0:
        cross_3d = np.cross(v[:3], b_3d)
        F[:3] += beltrami_damp_factor * cross_3d / b_norm
    # Toroidal moment contribution (3D)
    tm = toroidal_moment(pos_values, J)
    cross_tm = np.cross(tm, v[:3])
    F[:3] += 0.1 * cross_tm # Pad to 4D implicitly (F[3]=0)
    # Alfvén wave damping
    rho = 1.0 # Assume unit density - fixed if missing
    v_A = b_norm / np.sqrt(mu0 * rho)
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-8:
        F -= 0.05 * (v_norm - v_A) * v / v_norm
    return F # No clip - test natural
# New: Function to split traj on wrap steps (reusable)
def split_traj_on_wraps(traj, wrap_steps, start_step=0, end_step=None):
    if end_step is None:
        end_step = len(traj)
    # Filter wraps in range (global steps)
    relevant_wraps = sorted(set(w for w in wrap_steps if start_step <= w < end_step)) # Unique, sorted
    if not relevant_wraps:
        return [traj] # No splits, whole as list
    segments = []
    seg_start = 0
    for w in relevant_wraps:
        # Omit connect: Take up to w (pre-wrap), start new after w
        if seg_start < w - start_step: # Valid seg
            segments.append(traj[seg_start : w - start_step])
        seg_start = (w - start_step) + 1 # Skip the jump line
    # Last seg
    if seg_start < len(traj):
        segments.append(traj[seg_start:])
    return [seg for seg in segments if len(seg) > 1] # Drop tiny
# Initialize fields
B = np.random.normal(0, 0.1, (N, 4))
J = np.random.normal(0, central_I, (N, 4))
psi = np.ones(N, dtype=complex) / np.sqrt(N)
# Particle initialization
positions = []
velocities = []
cur_nodes = [None] * num_particles # Fixed: Full list of Nones
sf = particle_density_scale ** (1.0 / 3.0)  # Scale factor for 3D effective volume fraction
for p in range(num_particles):
    theta = random.uniform(0, 2*np.pi)
    phi = random.uniform(0, np.pi)
    r = random.uniform(0.4*L * sf, 0.6*L * sf)
    dir_vec = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi), 0.0])
    dir_vec /= np.linalg.norm(dir_vec) + 1e-20
    x = np.array([L/2, L/2, L/2, L/2]) + r * dir_vec
    radial = dir_vec
    tang_dir = np.random.randn(4)
    tang_dir -= np.dot(tang_dir, radial) * radial / np.linalg.norm(radial)**2
    tang_dir /= np.linalg.norm(tang_dir) + 1e-20
    v_tang = np.sqrt(force_scale * v_tang_scale / r) * 0.5
    swirl = swirl_strength * np.random.uniform(-1,1)
    v = v_tang * tang_dir + swirl * radial
    positions.append(x % period)
    velocities.append(v)
I_target = defaultdict(float)
for edge in G.edges():
    I_target[edge] = random.uniform(-1,1) * central_I
trajs = [[] for _ in range(num_particles)]
avg_speed_history = []
helicity_history = []
wrap_steps = [[] for _ in range(num_particles)] # New: List of steps where wrap occurred for each particle
freeze_counter = 0
last_avg_speed = 0.0
start_time = time.time()
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
json_file = os.path.join(results_dir, f"qad_{timestamp}_data.jsonl")
# Manual serializable param dict
param_dict = {
    "code_version": code_version,
    "mode": mode,
    "macro_optimizations": macro_optimizations,
    "random_seed": random_seed,
    "force_scale": force_scale,
    "base_spacing": base_spacing,
    "layers": layers,
    "I_particle": I_particle,
    "mu0": mu0,
    "v_tang_scale": v_tang_scale,
    "n_pert": n_pert,
    "t_max": t_max,
    "dt": dt,
    "Theta_base": Theta_base,
    "quantum_blend": quantum_blend,
    "hbar_sim": hbar_sim,
    "num_particles": num_particles,
    "c_lattice": c_lattice,
    "progress_every": progress_every,
    "central_I": central_I,
    "swirl_strength": swirl_strength,
    "freeze_threshold": freeze_threshold,
    "freeze_patience": freeze_patience,
    "beltrami_damp_factor": beltrami_damp_factor,
    "downsample_interval": downsample_interval,
    "viscosity_factor": viscosity_factor,
    "epsilon_factor": epsilon_factor,
    "fractal_exp": fractal_exp,
    "particle_density_scale": particle_density_scale
}
with open(json_file, 'w') as f:
    json.dump({"parameters": param_dict}, f)
    f.write('\n')
total_steps = int(t_max / dt)
total_steps_minus_one = total_steps - 1
traj_checkpoints = [[] for _ in range(num_particles)] # Temp for chunking trajs
wrap_checkpoints = [[] for _ in range(num_particles)] # New: For chunking wrap_steps
last_checkpoint_step = -1 # Track last dump step
next_progress = progress_every - 1 # initialise loop so it runs from 0 to N-1 eg 0 to 99
##############################################################################################################
#  MAIN SIMUALATION LOOP
##############################################################################################################
print("\n" + "="*50)
print(f"QAD SIMULATION v{code_version} | N={N} | Particles={num_particles}")
print("="*50)
for step in range(total_steps):
    # Evolve B and J fields (simplified MHD)
    curl_B = np.zeros((N, 4))
    for i, node in enumerate(node_list):
        nb_list = neighbors[node]
        deg = len(nb_list)
        if deg > 0:
            for dim in range(4):
                curl_B[i, dim] = sum(np.sign(pos_values[node_list.index(nb)][dim] - pos_values[i][dim]) * B[node_list.index(nb), dim] for nb in nb_list) / deg
    J += dt * (curl_B - mu0 * J) / mu0 # Resistive term
    curl_J = np.zeros((N, 4))
    for i in range(N):
        nb_list = neighbors[node_list[i]]
        deg = len(nb_list)
        if deg > 0:
            for dim in range(4):
                curl_J[i, dim] = sum(np.sign(pos_values[node_list.index(nb)][dim] - pos_values[i][dim]) * J[node_list.index(nb), dim] for nb in nb_list) / deg
    B += dt * curl_J
    # Evolve psi (discretized Schrödinger)
    lap_psi = graph_laplacian_real(psi.real) + 1j * graph_laplacian_real(psi.imag)
    psi += -1j * (hbar_sim / (2 * I_particle)) * lap_psi * dt
    norm = np.sum(np.abs(psi)**2)
    if norm > 0:
        psi /= np.sqrt(norm)
    # Particle updates with fixed Velocity Verlet
    for p in range(num_particles):
        x = positions[p]
        v = velocities[p]
        query_point = x % period
        _, cur_idx = tree.query(query_point)
        cur_node = node_list[cur_idx]
        I_nb = [I_target[(cur_node, nb) if (cur_node, nb) in I_target else (nb, cur_node)] for nb in neighbors[cur_node]]
        F = ampere_force(x, v, cur_node, I_nb, B, J, psi, p)
        a_mag = np.linalg.norm(F) / I_particle
        m_eff = qi_mass(a_mag)
        a_current = F / max(m_eff, 1e-12)
        a_current = np.nan_to_num(a_current)
        # a_current = np.clip(a_current, -1e10, 1e10) # Uncomment if needed
        v_half = v + 0.5 * a_current * dt * force_scale
        x_new = x + v_half * dt
        # New: Detect wrap before wrapping
        if np.any(x_new < 0) or np.any(x_new >= period):
            wrap_steps[p].append(step)
        query_point_new = x_new % period
        _, cur_idx_new = tree.query(query_point_new)
        cur_node_new = node_list[cur_idx_new]
        I_nb_new = [I_target[(cur_node_new, nb) if (cur_node_new, nb) in I_target else (nb, cur_node_new)] for nb in neighbors[cur_node_new]]
        new_F = ampere_force(x_new, v_half, cur_node_new, I_nb_new, B, J, psi, p)
        new_a_mag = np.linalg.norm(new_F) / I_particle
        new_m_eff = qi_mass(new_a_mag)
        a_new = new_F / max(new_m_eff, 1e-12)
        a_new = np.nan_to_num(a_new)
        # a_new = np.clip(a_new, -1e10, 1e10) # Uncomment if needed
        v_new = v_half + 0.5 * a_new * dt * force_scale
        if quantum_blend > 0:
            grad_S = compute_guidance(cur_node_new, psi)
            v_new = (1 - quantum_blend) * v_new + quantum_blend * grad_S
        # v_new = np.clip(v_new, -c_lattice/10, c_lattice/10) # Uncomment if needed
        v_new *= (1 - viscosity_factor) # Post-update damp
        positions[p] = x_new % period
        velocities[p] = v_new
        cur_nodes[p] = cur_node_new # Fixed: Update after full commit, for next step's consistency
    if step % downsample_interval == 0:
        for p in range(num_particles):
            trajs[p].append(list(positions[p]) + list(velocities[p]))
    avg_speed = np.mean([np.linalg.norm(velocities[p]) for p in range(num_particles)])
    avg_speed_history.append(avg_speed)
    helicity = compute_helicity(B, J)
    helicity_history.append(helicity)
    speed_change = abs(avg_speed - last_avg_speed)
    if speed_change < freeze_threshold:
        freeze_counter += 1
    else:
        freeze_counter = 0
    last_avg_speed = avg_speed
    if freeze_counter >= freeze_patience:
        print(f"Simulation frozen at step {step}. Speed stabilized.")
        break
    if step == 0:
        print(f"Step {step + 1}/{total_steps} | Avg speed: {avg_speed:.3f} | Helicity: {helicity_history[-1]:.3f} | Elapsed: {time.time()-start_time:.1f}s")
    if step == next_progress or step == total_steps_minus_one:
        print(f"Step {step + 1}/{total_steps} | Avg speed: {avg_speed:.3f} | Helicity: {helicity_history[-1]:.3f} | Elapsed: {time.time()-start_time:.1f}s")
        chunk_data = {
            "chunk_start_step": last_checkpoint_step + 1,
            "chunk_end_step": step,
            "trajectories_chunk": {f"particle_{p}": np.array(trajs[p][len(traj_checkpoints[p]):]).tolist() for p in range(num_particles)},
            "helicity_chunk": helicity_history[last_checkpoint_step + 1:],
            "avg_speed_chunk": avg_speed_history[last_checkpoint_step + 1:],
            "wrap_steps_chunk": {f"particle_{p}": wrap_steps[p][len(wrap_checkpoints[p]):] for p in range(num_particles)} # New: Chunk wrap steps
        }
        # Append the chunk
        with open(json_file, "a") as f:
            json.dump(chunk_data, f)
            f.write("\n")
        # Update checkpoints
        for p in range(num_particles):
            traj_checkpoints[p] = trajs[p][:] # Snapshot current
            wrap_checkpoints[p] = wrap_steps[p][:] # New: Snapshot wrap_steps
        last_checkpoint_step = step
        next_progress += progress_every
    last_step = step * dt
# New: Split trajs on wraps for clean plotting
# wrap_steps = {p: [] for p in range(num_particles)} # If not logged; else load from your update
# Assume logged in sim loop; if not, add detection as fallback
for p in range(num_particles):
    traj_full = np.array(trajs[p])
    segments = split_traj_on_wraps(traj_full[:, :3], wrap_steps[p]) # :3 for 3D
    # Plot segments instead of full
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.viridis(np.linspace(0,1,num_particles))
for pp in range(num_particles): # Loop all for multi
    segs_p = split_traj_on_wraps(np.array(trajs[pp])[:, :3], wrap_steps[pp])
    for seg in segs_p:
        ax.plot(seg[:,0], seg[:,1], seg[:,2] + 0.1 * seg[:,3] if len(seg[0]) > 3 else seg[:,2], lw=1.5, color=colors[pp], alpha=0.8)
ax.set_title(f"Particle Trajectories 3D | {N} Nodes | {total_steps} Steps | Seed {random_seed} | Density {particle_density_scale}")
param_text = f"force_scale={force_scale} | quantum_blend={quantum_blend} | v_tang_scale={v_tang_scale} | Theta_base={Theta_base}\nlayers={layers} | num_particles={num_particles} | beltrami_damp_factor={beltrami_damp_factor} | viscosity_factor={viscosity_factor}\nbase_spacing={base_spacing} | t_max={t_max} | downsample_int={downsample_interval} | swirl_strength = {swirl_strength} | V{code_version}"
ax.text2D(0.05, 0.99, param_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
plt.savefig(os.path.join(results_dir, f"qad_{timestamp}_orbits_3d.png"))
plt.close()
# Similar for XY (plot segs)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
# colors = plt.cm.viridis(np.linspace(0,1,num_particles))
for pp in range(num_particles): # Loop all for multi
    segs_p = split_traj_on_wraps(np.array(trajs[pp])[:, :3], wrap_steps[pp])
    for seg in segs_p:
        ax.plot(seg[:,0], seg[:,1], lw=1.5, color=colors[pp], alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f"Particle Trajectories XY | {N} Nodes | {total_steps} Steps | Seed {random_seed} | Density {particle_density_scale}")
ax.set_aspect('equal')
ax.text(0.05, 0.99, param_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
plt.savefig(os.path.join(results_dir, f"qad_{timestamp}_orbits_xy.png"))
plt.close()
print(f"\nTotal runtime: {time.time() - start_time:.1f} seconds")
print(f"\nSettings, speed, helicity and trajectories saved to {json_file}\n")
