# QAD-simulations
Quantum Aether Dynamics simulations - created and developed starting in November 2025.

### Running qad_simulation.py will produce: <br>
- a jsonl file that contains details of the parameters used in the simulation, as well as the particle locations, velocity and trajectories
- a 3d plot of the complete trajectories of all particles
- an XY plot of the trajectories of all particles

### Running qad_analysis.py will produce an analysis folder containing: <br>
- Plots and CSVs of avg speed, helicty, mean particle distace, alfven speedf, curvature and sphericity
- Plots of the Peaks of activity, and a manual plot - last 30 frames of simulation if unselected

### Runnting qad_animation.py will produce: <br>
- a mp4 video file with all or selected frames from the jsonl file chosen

<br>
<br>

### Code for QAD Particle paper: "Spontaneous Emergence of the Standard-Model Particle Spectrum from Classical Electrodynamics on a Discrete Lattice" <br>

Full code that produced the results in the paper: <br>
[FULL BRAID SIMUATION](2025/qad_braid_simulation.py)

There is also a simpllified version for easier compute: <br>
[SIMPLIFIED BRAID SIMUATION](2025/qad_braid_simple.py)

For revised Paper 2 we compare a cubic lattice to a hex lattice, this is the code used in the paper for a Hex lattics: <br>
[HEX BRAID SIMUATION](2025/qad_braid_hex.py)
