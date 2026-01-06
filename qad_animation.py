import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
import imageio.v2 as imageio
import os

# V3 ADDS ZOOM MODES: FIXED, MANUAL, AUTO
# V4 uses the "manual_steps" parameter for ALL modes - also fixed bug on wrap slit

json_data = 'qad_2026-01-06_01-03-28_data.jsonl'  # Replace with your JSONL dat file in qad_simulation_results folder

results_dir = "qad_simulation_results"
temp_dir = "qad_temp_frames"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

json_name = json_data.removesuffix('_data.jsonl')
json_fle = os.path.join(results_dir, json_data)

# Mode selection: 'fixed' (full plot), 'manual' (user axes/step range), 'auto' (dynamic zoom after threshold)
mode = 'manual'  # Change here: 'fixed', 'manual', 'auto'
# For manual: Define axes mins/maxes (x,y,z), step range (start, end)
manual_min = [88.2, 45.0, 34.0]
manual_max = [89.4, 47.4, 35.5]
manual_steps = (0, 0)  # Inclusive ***** APPLIES TO ALL MODES !!!!!! ***** Use (0, 0) for ALL
# For auto: After this step, start zooming; target fill ~80% of axes
auto_start_step = 10
auto_target_fill = 0.8  # Fraction of axes to occupy

# Reusable split function
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

# Load full trajectories and wraps
def load_trajectories(json_file):
    trajs = {}
    wraps = {}
    with open(json_file, 'r') as f:
        for line in f:
            data = json.loads(line)
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
    return trajs, wraps

trajs, wraps = load_trajectories(json_file)
keys = sorted(trajs.keys())
num_particles = len(keys)

# Fixed axes from full data if fixed/manual
all_pos = np.vstack([np.array(trajs[p])[:, :3] for p in keys])
xmin_f, ymin_f, zmin_f = all_pos.min(axis=0)
xmax_f, ymax_f, zmax_f = all_pos.max(axis=0)
buffer = 0.1 * max([xmax_f - xmin_f, ymax_f - ymin_f, zmax_f - zmin_f])
xmin_f -= buffer
xmax_f += buffer
ymin_f -= buffer
ymax_f += buffer
zmin_f -= buffer
zmax_f += buffer

# Colors
colors = plt.cm.viridis(np.linspace(0, 1, num_particles))

# Animation params
full_steps = len(trajs[keys[0]])
frame_every = 1 # was 1 - uswd 25 for a long run
trail_steps = 10 # was 10 when interval = 1, used 100 for long run
num_fade_segments = 5 # was 5 when interval = 1, used 25 for long run
min_points = 1 # was 4 when interval = 1, use 20 for long run
frame_files = []

# Slice steps using 'manual_steps' variable if present
start_frame = min_points
end_frame = full_steps + 1

if manual_steps and len(manual_steps) == 2:
    ms_start, ms_end = manual_steps
    # Validate: meaningful range, non-negative start, logical order
    if isinstance(ms_start, int) and isinstance(ms_end, int):
        if ms_start >= 0 and ms_end >= ms_start - 1 and (ms_start != 0 or ms_end != 0):
            start_frame = ms_start
            end_frame = min(ms_end + 1, full_steps + 1)
            # Final safety: ensure non-empty range
            if start_frame >= end_frame:
                print("Warning: Manual range results in empty slice. Using auto range.")
                start_frame = min_points
                end_frame = full_steps + 1

# Initial axes for auto (start with fixed)
xmin, xmax = xmin_f, xmax_f
ymin, ymax = ymin_f, ymax_f
zmin, zmax = zmin_f, zmax_f

# Generate frames
for frame_idx, end in enumerate(range(start_frame, end_frame, frame_every)):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    current_pos = []  # For auto zoom
    for k, p_key in enumerate(keys):
        traj_full = np.array(trajs[p_key])[:end, :3]
        if len(traj_full) > trail_steps:
            traj_part = traj_full[-trail_steps:]
        else:
            traj_part = traj_full
        window_start_step = end - len(traj_part)
        wrap_steps_p = wraps.get(p_key, [])
        relevant_wraps = [w for w in wrap_steps_p if window_start_step <= w < end]
        segments = split_traj_on_wraps(traj_part, relevant_wraps, window_start_step, end)
        for seg_idx, seg in enumerate(segments):
            base_alpha = 1.0 - 0.5 * (seg_idx / max(1, len(segments) - 1)) if len(segments) > 1 else 1.0
            if len(seg) > 3:
                try:
                    diff = np.diff(seg, axis=0)
                    cum_speed = np.insert(np.cumsum(np.linalg.norm(diff, axis=1)), 0, 0)
                    tck, u = splprep(seg.T, u=cum_speed, s=0)
                    u_fine = np.linspace(0, cum_speed[-1], len(seg) * 5)
                    smooth = np.array(splev(u_fine, tck)).T
                except Exception as e:
                    print(f"Spline failed for {p_key} seg {seg_idx} at end={end}: {e}")
                    smooth = seg
            else:
                smooth = seg
            if len(smooth) > num_fade_segments * 2:
                seg_size = len(smooth) // num_fade_segments
                for s in range(num_fade_segments):
                    start = s * seg_size
                    end_s = (s + 1) * seg_size if s < num_fade_segments - 1 else len(smooth)
                    alpha = base_alpha * (0.2 + 0.8 * (s / (num_fade_segments - 1)))
                    ax.plot(smooth[start:end_s, 0], smooth[start:end_s, 1], smooth[start:end_s, 2],
                            color=colors[k], alpha=alpha, lw=1.5)
            else:
                ax.plot(smooth[:, 0], smooth[:, 1], smooth[:, 2], color=colors[k], alpha=base_alpha, lw=1.5)
            if seg_idx == len(segments) - 1 and len(smooth) > 0:
                ax.scatter(smooth[-1, 0], smooth[-1, 1], smooth[-1, 2], color=colors[k], s=20)
                current_pos.append(smooth[-1])
    ax.set_title(f"Smoothed 3D Trajectories with Trails: Steps {end - min(end, trail_steps)}-{end-1}")
    # Mode-specific axes
    if mode == 'fixed':
        ax.set_xlim([xmin_f, xmax_f])
        ax.set_ylim([ymin_f, ymax_f])
        ax.set_zlim([zmin_f, zmax_f])
    elif mode == 'manual':
        ax.set_xlim([manual_min[0], manual_max[0]])
        ax.set_ylim([manual_min[1], manual_max[1]])
        ax.set_zlim([manual_min[2], manual_max[2]])
    elif mode == 'auto' and end >= auto_start_step:
        if current_pos:
            current_pos = np.array(current_pos)
            xmin_n, ymin_n, zmin_n = current_pos.min(axis=0)
            xmax_n, ymax_n, zmax_n = current_pos.max(axis=0)
            spread = np.array([xmax_n - xmin_n, ymax_n - ymin_n, zmax_n - zmin_n])
            max_spread = np.max(spread)
            if max_spread > 0:
                target_spread = max_spread / auto_target_fill
                pad = (target_spread - spread) / 2
                xmin = xmin_n - pad[0]
                xmax = xmax_n + pad[0]
                ymin = ymin_n - pad[1]
                ymax = ymax_n + pad[1]
                zmin = zmin_n - pad[2]
                zmax = zmax_n + pad[2]
                # Smooth transition (lerp from previous)
                if 'prev_xmin' in globals():
                    xmin = 0.9 * prev_xmin + 0.1 * xmin
                    xmax = 0.9 * prev_xmax + 0.1 * xmax
                    ymin = 0.9 * prev_ymin + 0.1 * ymin
                    ymax = 0.9 * prev_ymax + 0.1 * ymax
                    zmin = 0.9 * prev_zmin + 0.1 * zmin
                    zmax = 0.9 * prev_zmax + 0.1 * zmax
                prev_xmin, prev_xmax = xmin, xmax
                prev_ymin, prev_ymax = ymin, ymax
                prev_zmin, prev_zmax = zmin, zmax
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_zlim([zmin, zmax])
    ax.view_init(elev=30, azim=45)
    frame_file = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
    plt.savefig(frame_file)
    frame_files.append(frame_file)
    plt.close(fig)

# Compile into MP4
images = [imageio.imread(f) for f in frame_files]
imageio.mimsave(os.path.join(results_dir, f'{json_name}_animation.mp4', images, fps=10)

# Clean up
for f in frame_files:
    os.remove(f)
print(f"Animation saved as '{json_name}_animation.mp4'")
