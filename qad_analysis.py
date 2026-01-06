import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
import pandas as pd

# qad_analysis_9.py - Added zoomed plots for persistent co-orbitals
# 9b adds X,Y,Z labels to axes - and comments out co-orbit plots - there were hundreds of them!!
# 9b also fixes bug on wrap split
# 9c adds alfen speed calculation
# 9d saves final chunk as seprate "final_chunk" file - also exports "info" file

json_data = 'qad_2026-01-06_01-03-28_data.jsonl'  # Replace with your file in qad_simulation_results folder
manual_plot = True
manual_start, manual_end = 69970, 69999  # use 0, 0 for auto - which will show last 30 frames
measure_alfven_speed = True

# Create results folder if not exists
results_dir = "qad_simulation_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

json_name = json_data.removesuffix('_data.jsonl')
json_file = os.path.join(results_dir, json_data)
analysis_dir = os.path.join(results_dir, f'{json_name}_analysis')
os.makedirs(analysis_dir, exist_ok=True)


def mean_curvature(traj):
    if len(traj) < 3:
        return 0.0
    diff1 = np.diff(traj, axis=0)
    diff2 = np.diff(diff1, axis=0)
    kappa = np.linalg.norm(np.cross(diff1[:-1], diff2), axis=1) / (np.linalg.norm(diff1[:-1], axis=1)**3 + 1e-6)
    return np.mean(kappa)

def estimate_alfven_speed(pos, vel, c_lattice, mu0=0.5, mass_per_particle=1.0, k_neighbors=5):
    from scipy.spatial import KDTree
    from scipy.spatial.distance import pdist
    # Local density rho: volume from k-NN
    tree = KDTree(pos)
    dists, _ = tree.query(pos, k=k_neighbors+1)
    mean_radius = np.mean(dists[:,1:], axis=1)
    mean_radius = np.maximum(mean_radius, 1e-3)  # Prevent zero-volume
    local_vol = (4/3) * np.pi * mean_radius**3 / k_neighbors
    rho = mass_per_particle / local_vol
    rho_mean = np.mean(rho)
    rho_mean = min(rho_mean, 1e6)  # Cap density to avoid unphysical infinities (tune based on density_scale=1e-6)
    # Effective J ~ rho * v_mean (velocity as current proxy)
    v_norms = np.linalg.norm(vel, axis=1)
    v_mean = np.mean(v_norms)
    J_mean = rho_mean * v_mean
    # Effective B ~ mu0 J / (2 pi r_mean) for filament
    r_mean = np.mean(pdist(pos)) / len(pos)  # normalized avg separation
    r_mean = max(r_mean, 1e-3)  # Physical min scale, e.g., lattice spacing/50
    B_eff = mu0 * J_mean / (2 * np.pi * (r_mean + 1e-6))
    # hall_term = (mass_per_particle * J_mean) / (1.0 * rho_mean + 1e-6)  # e~1 proxy
    v_A = B_eff / np.sqrt(mu0 * rho_mean + 1e-6) # + hall_term
    v_A /= c_lattice  # Normalize to relative speed
    return v_A

def sphericity_proxy(points):
    if len(points) < 2:
        return 0.0
    center = np.mean(points, axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    return np.std(dists) / (np.mean(dists) + 1e-6)

def gauss_linking_number(traj1, traj2, dim=3, downsample=1, close_loops=False, min_distance=1e-6):
    traj1 = np.array(traj1)[:, :dim]
    traj2 = np.array(traj2)[:, :dim]
    if close_loops:
        if len(traj1) > 1 and len(traj2) > 1:
            traj1 = np.vstack([traj1, traj1[0]])
            traj2 = np.vstack([traj2, traj2[0]])
    traj1 = traj1[::downsample]
    traj2 = traj2[::downsample]
    if len(traj1) < 2 or len(traj2) < 2:
        return 0.0
    r1 = traj1[:-1][:, None, :]
    dr1 = (traj1[1:] - traj1[:-1])[:, None, :]
    r2 = traj2[:-1][None, :, :]
    dr2 = (traj2[1:] - traj2[:-1])[None, :, :]
    r12 = r1 - r2
    denom = np.linalg.norm(r12, axis=2) ** 3
    cross = np.cross(dr1, dr2)
    dot = np.sum(cross * r12, axis=2)
    valid = denom > min_distance
    increments = np.zeros_like(denom)
    increments[valid] = dot[valid] / denom[valid]
    link = np.sum(increments) / (4 * np.pi)
    return abs(link)

def self_writhe(traj, dim=3, downsample=1, close_loops=False, eps=1e-3, min_distance=1e-6):
    traj_pert = np.array(traj) + np.random.normal(0, eps, np.array(traj).shape)
    return gauss_linking_number(traj, traj_pert, dim, downsample, close_loops, min_distance)

def compute_per_chunk_linking(trajs, chunk_size, dim=3, downsample=1, close_loops=False, min_distance=1e-6):
    keys = list(trajs.keys())
    if len(keys) < 2:
        print("Need at least 2 particles for linking; skipping.")
        return np.array([]), np.array([])
    traj_fulls = {k: np.array(trajs[k]) for k in keys}
    links = []
    chunk_starts = []
    n_pairs = len(keys) * (len(keys) - 1) / 2
    for start in range(0, len(traj_fulls[keys[0]]), chunk_size):
        end = min(start + chunk_size, len(traj_fulls[keys[0]]))
        pair_links = []
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                traj_i = traj_fulls[keys[i]][start:end]
                traj_j = traj_fulls[keys[j]][start:end]
                link = gauss_linking_number(traj_i, traj_j, dim, downsample, close_loops, min_distance)
                pair_links.append(link)
        avg_link = np.mean(pair_links) if pair_links else 0.0
        links.append(avg_link)
        chunk_starts.append(start)
    return np.array(chunk_starts), np.array(links)

def split_traj_on_wraps(traj, wrap_steps, start_step=0, end_step=None):
    if len(traj) == 0:  # New: Handle empty traj
        return []
    if end_step is None:
        end_step = len(traj)
    relevant_wraps = sorted(set(w for w in wrap_steps if start_step <= w < end_step))
    if not relevant_wraps:
        return [traj]
    segments = []
    seg_start = 0
    for w in relevant_wraps:
        if seg_start < w - start_step:
            segments.append(traj[seg_start : w - start_step])
        seg_start = (w - start_step) + 1
    if seg_start < len(traj):
        segments.append(traj[seg_start:])
    return [seg for seg in segments if len(seg) > 1]

def unwrap_traj(traj, period):
    unwrapped = traj.copy()
    for dim in range(traj.shape[1]):
        diff = np.diff(traj[:, dim])
        jumps = np.abs(diff) > period[dim] / 2
        cum_shift = np.cumsum(jumps * np.sign(diff) * period[dim])
        unwrapped[1:, dim] -= cum_shift
    return unwrapped

# Load JSONL
trajs = {}
vels = {}
wraps = {}
helicity = []
chunk_datas = []  # New 9d: List to hold full chunk dicts
period = None
with open(json_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        if 'parameters' in data:
            params = data['parameters']
            print(f"Loaded params: num_particles={params['num_particles']}")
            layers = params.get('layers', 10)
            base_spacing = params.get('base_spacing', 50.0)
            c_lattice = params.get('c_lattice', 3000)
            L = base_spacing * layers ** 0.25
            period = [L, L, L, L]  # 4D
        if 'trajectories_chunk' in data:
            for p, t_chunk in data['trajectories_chunk'].items():
                if p not in trajs:
                    trajs[p] = []
                trajs[p].extend(t_chunk)
        if 'wrap_steps_chunk' in data:
            for p, w_chunk in data['wrap_steps_chunk'].items():
                if p not in wraps:
                    wraps[p] = []
                wraps[p].extend(w_chunk)
        if 'helicity_chunk' in data:
            helicity.extend(data['helicity_chunk'])
        if 'trajectories_chunk' in data:
            chunk_datas.append(data)  # Store the full chunk dict

# Export final chunk if multiple
if chunk_datas:
    final_chunk = chunk_datas[-1]  # Last chunk dict
    final_chunk_file = os.path.join(analysis_dir, f'{json_name}_final_chunk.jsonl')
    with open(final_chunk_file, 'w') as ff:
        ff.write(json.dumps({'parameters': params}) + '\n')  # Parameters first
        ff.write(json.dumps(final_chunk) + '\n')  # Then the chunk
    print(f"Exported final chunk to {final_chunk_file}")
else:
    print("No chunks found; skipping final chunk export.")

print("Chunks joined; full trajectories and wraps loaded.")

keys = list(trajs.keys())
if not keys or all(len(trajs[k]) == 0 for k in keys):
    print("No trajectory data loaded; skipping all analysis.")
else:
    trajs = {k: np.array(trajs[k]) for k in keys}
    # Assuming velocities are concatenated in trajs as [pos + vel], split if needed
    # If velocities are separate, load vels similarly; here assuming trajs has [x,y,z,w,vx,vy,vz,vw]

# Export info: parameters + initial/final pos/vel
if keys and all(len(trajs[k]) > 0 for k in keys):
    initial = {k: list(trajs[k][0]) for k in keys}  # Full initial vector [pos + vel + ...]
    final = {k: list(trajs[k][-1]) for k in keys}  # Full final vector
    info_file = os.path.join(analysis_dir, f'{json_name}_info.jsonl')
    with open(info_file, 'w') as fi:
        fi.write(json.dumps({'parameters': params}) + '\n')
        fi.write(json.dumps({'initial': initial}) + '\n')
        fi.write(json.dumps({'final': final}) + '\n')
    print(f"Exported info to {info_file}")
else:
    print("No trajectory data; skipping info export.")

# Compute per-chunk linking
chunk_size = 50
chunk_starts, chunk_links = compute_per_chunk_linking(trajs, chunk_size, downsample=1, close_loops=True)

# Overview plot of linking
if len(chunk_links) > 0:
    plt.figure(figsize=(8, 4))
    plt.plot(chunk_starts, chunk_links, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Average Linking Number')
    plt.title(f'{json_name} - Average Linking vs. Step')
    plt.savefig(os.path.join(analysis_dir, f'{json_name}_linking_overview.png'))
    plt.close()
    print(f"Average linking across all chunks: {np.mean(chunk_links):.2f}")

# Time-series metrics per chunk (mean and per-particle)
chunk_writhes_mean = []
chunk_curvs_mean = []
chunk_sphs_mean = []
chunk_alfvens_mean = []  # For Alfvén analogs if flagged
per_particle_metrics = {k: {'writhe': [], 'curv': [], 'sph': []} for k in keys}
for start in chunk_starts:
    end = min(start + chunk_size, len(trajs[keys[0]]))
    traj_chunks = {k: trajs[k][start:end] for k in keys}  # Full array per particle
    writhes_chunk = []
    curvs_chunk = []
    sphs_chunk = []
    for k in keys:
        traj_ch = traj_chunks[k][:, :3]  # Pos only for these metrics
        if len(traj_ch) > 1:
            wr = self_writhe(traj_ch)
            per_particle_metrics[k]['writhe'].append(wr)
            writhes_chunk.append(wr)
        if len(traj_ch) > 2:
            curv = mean_curvature(traj_ch)
            per_particle_metrics[k]['curv'].append(curv)
            curvs_chunk.append(curv)
        if len(traj_ch) > 1:
            sph = sphericity_proxy(traj_ch)
            per_particle_metrics[k]['sph'].append(sph)
            sphs_chunk.append(sph)
    chunk_writhes_mean.append(np.mean(writhes_chunk) if writhes_chunk else 0.0)
    chunk_curvs_mean.append(np.mean(curvs_chunk) if curvs_chunk else 0.0)
    chunk_sphs_mean.append(np.mean(sphs_chunk) if sphs_chunk else 0.0)
    
    if measure_alfven_speed:
        alfvens_chunk = []
        # Aggregate all pos/vel in chunk
        all_pos = np.vstack([traj_chunks[k][:, :3] for k in keys if len(traj_chunks[k]) > 0])
        all_vel = np.vstack([traj_chunks[k][:, 3:6] for k in keys if len(traj_chunks[k]) > 0])  # Assuming [pos, vel]
        if len(all_pos) > 1 and len(all_vel) > 1:
            alfven_chunk = estimate_alfven_speed(all_pos, all_vel, c_lattice)
            alfvens_chunk.append(alfven_chunk)
        chunk_alfvens_mean.append(np.mean(alfvens_chunk) if alfvens_chunk else 0.0)

# Plot mean time-series metrics
plt.figure(figsize=(10, 6))
plt.plot(chunk_starts, chunk_writhes_mean, label='Mean Writhe')
plt.plot(chunk_starts, chunk_curvs_mean, label='Mean Curvature')
plt.plot(chunk_starts, chunk_sphs_mean, label='Mean Sphericity')
if measure_alfven_speed:
    plt.plot(chunk_starts, chunk_alfvens_mean, label='Mean Alfvén Speed Analog')
plt.xlabel('Step')
plt.ylabel('Value')
plt.title(f'{json_name} - Time-Series Metrics (Means)')
plt.legend()
plt.savefig(os.path.join(analysis_dir, f'{json_name}_time_series_metrics_means.png'))
plt.close()

# Table for mean time-series metrics
metrics_df = pd.DataFrame({
    'Step Start': chunk_starts,
    'Mean Writhe': chunk_writhes_mean,
    'Mean Curvature': chunk_curvs_mean,
    'Mean Sphericity': chunk_sphs_mean
})
if measure_alfven_speed:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(chunk_starts, chunk_writhes_mean, label='Mean Writhe', color='blue')
    ax1.plot(chunk_starts, chunk_curvs_mean, label='Mean Curvature', color='orange')
    ax1.plot(chunk_starts, chunk_sphs_mean, label='Mean Sphericity', color='green')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Writhe / Curvature / Sphericity')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()  # Twin axis for Alfvén
    ax2.semilogy(chunk_starts, chunk_alfvens_mean, label='Mean Alfvén Speed Analog', color='red') # replace semilogy for plot for linear
    ax2.set_ylabel('Alfvén Speed (Normalized)')
    ax2.legend(loc='upper right')
    
    plt.title(f'{json_name} - Time-Series Metrics with Twin Axes')
    plt.savefig(os.path.join(analysis_dir, f'{json_name}_time_series_metrics_twin.png'))
    plt.close()

# Per-particle plots and CSVs (top 5 by variance for each metric)
n_top = 5
for metric in ['writhe', 'curv', 'sph']:
    variances = {k: np.var(per_particle_metrics[k][metric]) for k in keys if len(per_particle_metrics[k][metric]) > 0}
    top_keys = sorted(variances, key=variances.get, reverse=True)[:n_top]
    plt.figure(figsize=(10, 6))
    for k in top_keys:
        plt.plot(chunk_starts[:len(per_particle_metrics[k][metric])], per_particle_metrics[k][metric], label=k)
    plt.xlabel('Step')
    plt.ylabel(metric.capitalize())
    plt.title(f'{json_name} - Top {n_top} Particles by {metric} Variance')
    plt.legend()
    plt.savefig(os.path.join(analysis_dir, f'{json_name}_top_{metric}.png'))
    plt.close()
    df_per = pd.DataFrame({k: per_particle_metrics[k][metric] for k in keys})
    df_per['Step Start'] = chunk_starts[:df_per.shape[0]]
    df_per.to_csv(os.path.join(analysis_dir, f'{json_name}_{metric}_per_particle.csv'), index=False)

# Average/Min distances per chunk
avg_dists = []
min_dists = []
for start in chunk_starts:
    end = min(start + chunk_size, len(trajs[keys[0]]))
    pos_chunk = np.array([trajs[k][end-1][:3] for k in keys if len(trajs[k]) > end-1])
    if len(pos_chunk) > 1:
        dists = pdist(pos_chunk)
        avg_dists.append(np.mean(dists))
        min_dists.append(np.min(dists))
    else:
        avg_dists.append(0.0)
        min_dists.append(0.0)

# Plot avg pairwise dist
plt.figure(figsize=(8, 4))
plt.plot(chunk_starts, avg_dists, marker='o')
plt.xlabel('Step')
plt.ylabel('Average Pairwise Distance')
plt.title(f'{json_name} - Average Pairwise Distance vs. Step')
plt.savefig(os.path.join(analysis_dir, f'{json_name}_avg_pairwise_dist.png'))
plt.close()

# Table for distances
dist_df = pd.DataFrame({
    'Step Start': chunk_starts,
    'Avg Distance': avg_dists,
    'Min Distance': min_dists
})
dist_df.to_csv(os.path.join(analysis_dir, f'{json_name}_distances.csv'), index=False)

# Co-orbital detection: Persistent close pairs with high vel correlation, track active chunks
co_orbital_threshold_dist = 10.0
co_orbital_threshold_cos = 0.9
co_orbital_min_chunks = 3
pair_active_chunks = {}
for chunk_idx, start in enumerate(chunk_starts):
    end = min(start + chunk_size, len(trajs[keys[0]]))
    pos_chunk = np.array([trajs[k][end-1][:3] for k in keys if len(trajs[k]) > end-1])
    vel_chunk = np.array([trajs[k][end-1][3:6] for k in keys if len(trajs[k]) > end-1])  # Assuming vel in trajs
    tree = KDTree(pos_chunk)
    pairs = tree.query_pairs(co_orbital_threshold_dist)
    for a, b in pairs:
        vel_a = vel_chunk[a] / np.linalg.norm(vel_chunk[a]) if np.linalg.norm(vel_chunk[a]) > 0 else np.zeros(3)
        vel_b = vel_chunk[b] / np.linalg.norm(vel_chunk[b]) if np.linalg.norm(vel_chunk[b]) > 0 else np.zeros(3)
        cos_angle = np.abs(np.dot(vel_a, vel_b))
        if cos_angle > co_orbital_threshold_cos:
            pair = tuple(sorted((keys[a], keys[b])))
            if pair not in pair_active_chunks:
                pair_active_chunks[pair] = []
            pair_active_chunks[pair].append(chunk_idx)

persistent_pairs = [p for p, chunks in pair_active_chunks.items() if len(chunks) >= co_orbital_min_chunks]

# Comment out co-orbit plots as per 9b
# if persistent_pairs: ... (full block commented)

# Peak detection and zooming (reduced window)
if len(chunk_links) > 0:
    peaks, _ = find_peaks(chunk_links, prominence=50)
    print(f"Detected {len(peaks)} peaks at chunk indices: {peaks}")
    for idx in peaks:
        print(f"Peak at chunk {idx} (start {chunk_starts[idx]}): Linking {chunk_links[idx]:.2f}")
    peak_indices = peaks if len(peaks) > 0 else [np.argmax(chunk_links)]
    zoom_window = 15
    for i, peak_idx in enumerate(peak_indices):
        peak_start = chunk_starts[peak_idx]
        peak_end = min(peak_start + chunk_size, len(trajs[keys[0]]))
        zoom_start = max(peak_start - zoom_window, 0)
        zoom_end = min(peak_end + zoom_window, len(trajs[keys[0]]))
        traj_zooms = {k: unwrap_traj(trajs[k][zoom_start:zoom_end, :3], period[:3]) for k in keys}
        # Straight plot
        fig_straight = plt.figure(figsize=(10, 8))
        ax_straight = fig_straight.add_subplot(111, projection='3d')
        ax_straight.set_xlabel('X')
        ax_straight.set_ylabel('Y')
        ax_straight.set_zlabel('Z')
        for p_key in keys:
            traj_zoom = traj_zooms[p_key]
            wrap_steps_p = wraps.get(p_key, [])
            segments = split_traj_on_wraps(traj_zoom, wrap_steps_p, zoom_start, zoom_end)
            for seg in segments:
                ax_straight.plot(seg[:,0], seg[:,1], seg[:,2])
        ax_straight.set_title(f'{json_name} - Zoomed 3D Straight: Steps {zoom_start}-{zoom_end} (Peak {i+1})')
        plt.savefig(os.path.join(analysis_dir, f'{json_name}_peak_{i+1}_{zoom_start}-{zoom_end}_straight.png'))
        plt.close(fig_straight)
        # Curved plot
        fig_curved = plt.figure(figsize=(10, 8))
        ax_curved = fig_curved.add_subplot(111, projection='3d')
        ax_curved.set_xlabel('X')
        ax_curved.set_ylabel('Y')
        ax_curved.set_zlabel('Z')
        writhes = {}
        for p_key in keys:
            traj_zoom = traj_zooms[p_key]
            wrap_steps_p = wraps.get(p_key, [])
            segments = split_traj_on_wraps(traj_zoom, wrap_steps_p, zoom_start, zoom_end)
            seg_writhes = []
            for seg in segments:
                if len(seg) > 3:
                    try:
                        diff = np.diff(seg, axis=0)
                        cum_speed = np.insert(np.cumsum(np.linalg.norm(diff, axis=1)), 0, 0)
                        tck, u = splprep(seg.T, u=cum_speed, s=0)
                        u_fine = np.linspace(0, cum_speed[-1], len(seg)*5)
                        smooth = np.array(splev(u_fine, tck)).T
                        ax_curved.plot(smooth[:,0], smooth[:,1], smooth[:,2])
                        seg_writhes.append(self_writhe(smooth))
                    except Exception as e:
                        print(f"Spline failed for {p_key}: {e}")
                        ax_curved.plot(seg[:,0], seg[:,1], seg[:,2])
            writhes[p_key] = np.mean(seg_writhes) if seg_writhes else 0.0
            print(f"Zoom writhe {p_key}: {writhes[p_key]:.4f}")
        ax_curved.set_title(f'{json_name} - Zoomed 3D Curved: Steps {zoom_start}-{zoom_end} (Peak {i+1})')
        plt.savefig(os.path.join(analysis_dir, f'{json_name}_peak_{i+1}_{zoom_start}-{zoom_end}_curved.png'))
        plt.close(fig_curved)

# Clustering and zoomed plots (dynamic radius, reduced window for traj snippet)
cluster_threshold = 10.0
pos_final = np.array([trajs[k][-1][:3] for k in keys if len(trajs[k]) > 0])
tree = KDTree(pos_final)
pairs = tree.query_pairs(cluster_threshold)
if len(pairs) > 0:
    close_indices = set()
    for i, j in pairs:
        close_indices.add(i)
        close_indices.add(j)
    close_pos = pos_final[list(close_indices)]
    center = np.mean(close_pos, axis=0)
    max_dist = np.max(np.linalg.norm(close_pos - center, axis=1))
    radius = max(max_dist * 1.2, 5.0)
    # Zoomed plot (last 30 steps)
    cluster_zoom_steps = 30
    cluster_start = max(len(trajs[keys[0]]) - cluster_zoom_steps, 0)
    fig_cluster = plt.figure(figsize=(10, 8))
    ax_cluster = fig_cluster.add_subplot(111, projection='3d')
    ax_cluster.set_xlabel('X')
    ax_cluster.set_ylabel('Y')
    ax_cluster.set_zlabel('Z')
    ax_cluster.set_xlim(center[0] - radius, center[0] + radius)
    ax_cluster.set_ylim(center[1] - radius, center[1] + radius)
    ax_cluster.set_zlim(center[2] - radius, center[2] + radius)
    for idx in close_indices:
        p_key = keys[idx]
        traj_snip = unwrap_traj(trajs[p_key][cluster_start:, :3], period[:3])
        wrap_steps_p = wraps.get(p_key, [])
        segments = split_traj_on_wraps(traj_snip, wrap_steps_p, cluster_start)
        for seg in segments:
            ax_cluster.plot(seg[:,0], seg[:,1], seg[:,2])
    ax_cluster.set_title(f'{json_name} - Zoomed Cluster (Steps {cluster_start}-{len(trajs[keys[0]])-1}, Final Close Group)')
    plt.savefig(os.path.join(analysis_dir, f'{json_name}_cluster_{cluster_start}-{len(trajs[keys[0]])-1}_zoomed.png'))
    plt.close(fig_cluster)
else:
    print("No close clusters found.")

# Max peak metrics (summary only)
if len(chunk_links) > 0:
    max_peak_start = chunk_starts[np.argmax(chunk_links)]
else:
    max_peak_start = 0
max_zoom_start = max(max_peak_start - 15, 0)
max_zoom_end = min(max_peak_start + chunk_size, len(trajs[keys[0]]))
traj_maxs = {k: unwrap_traj(trajs[k][max_zoom_start:max_zoom_end, :3], period[:3]) for k in keys}
curvs = {}
sphs = {}
for p_key in keys:
    traj_max = traj_maxs[p_key]
    curvs[p_key] = mean_curvature(traj_max)
    sphs[p_key] = sphericity_proxy(traj_max)

# Summary
print(f"Max peak zoom summary: Mean Curvature {np.mean(list(curvs.values())):.4f}, Max Curvature {max(curvs.values()):.4f}")
print(f"Max peak zoom summary: Mean Sphericity {np.mean(list(sphs.values())):.4f}, Min Sphericity (most spherical) {min(sphs.values()):.4f}")

# Manual range example
if manual_plot:
    traj_zooms_man = {k: unwrap_traj(trajs[k][manual_start:manual_end, :3], period[:3]) for k in keys}
    fig_straight_man = plt.figure(figsize=(10, 8))
    ax_straight_man = fig_straight_man.add_subplot(111, projection='3d')
    ax_straight_man.set_xlabel('X')
    ax_straight_man.set_ylabel('Y')
    ax_straight_man.set_zlabel('Z')
    for p_key in keys:
        traj_man = traj_zooms_man[p_key]
        wrap_steps_p = wraps.get(p_key, [])
        segments = split_traj_on_wraps(traj_man, wrap_steps_p, manual_start, manual_end)
        for seg in segments:
            ax_straight_man.plot(seg[:,0], seg[:,1], seg[:,2])
    ax_straight_man.set_title(f'{json_name} - Manual 3D Straight: Steps {manual_start}-{manual_end}')
    plt.savefig(os.path.join(analysis_dir, f'{json_name}_manual_{manual_start}-{manual_end}_straight.png'))
    plt.close(fig_straight_man)
    fig_curved_man = plt.figure(figsize=(10, 8))
    ax_curved_man = fig_curved_man.add_subplot(111, projection='3d')
    ax_curved_man.set_xlabel('X')
    ax_curved_man.set_ylabel('Y')
    ax_curved_man.set_zlabel('Z')
    for p_key in keys:
        traj_man = traj_zooms_man[p_key]
        wrap_steps_p = wraps.get(p_key, [])
        segments = split_traj_on_wraps(traj_man, wrap_steps_p, manual_start, manual_end)
        for seg in segments:
            if len(seg) > 3:
                try:
                    diff = np.diff(seg, axis=0)
                    cum_speed = np.insert(np.cumsum(np.linalg.norm(diff, axis=1)), 0, 0)
                    tck, u = splprep(seg.T, u=cum_speed, s=0)
                    u_fine = np.linspace(0, cum_speed[-1], len(seg)*5)
                    smooth = np.array(splev(u_fine, tck)).T
                    ax_curved_man.plot(smooth[:,0], smooth[:,1], smooth[:,2])
                except Exception as e:
                    print(f"Spline failed for {p_key}: {e}")
                    ax_curved_man.plot(seg[:,0], seg[:,1], seg[:,2])
    ax_curved_man.set_title(f'{json_name} - Manual 3D Curved: Steps {manual_start}-{manual_end}')
    plt.savefig(os.path.join(analysis_dir, f'{json_name}_manual_{manual_start}-{manual_end}_curved.png'))
    plt.close(fig_curved_man)
