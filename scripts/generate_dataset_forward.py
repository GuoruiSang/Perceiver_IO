import mujoco
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import h5py
import os
from tqdm import tqdm
import math

def randomly_initialize_qpos_qvel_qacc(model, lim_qpos=1, lim_qvel=1):
    """
        Args: 
            model: The mujoco model.
        Returns:
            initial_qpos: (model.nq, ). A randomly sampled position.
            initial_qvel: (model.nv, ). A randomly sampled velocity
    """

    # Get the dimensions of qpos, qvel, and qcc
    qpos_dim = model.nq
    qvel_dim = model.nv

    # Uniformly sample initial qpos and qvel
    initial_qpos = np.random.uniform(low=-lim_qpos, high=lim_qpos, size=(qpos_dim, ))
    initial_qvel = np.random.uniform(low=-lim_qvel, high=lim_qvel, size=(qvel_dim, ))

    # In forward dynamics, qacc is an output, not an input state.
    # initial_qacc = np.zeros((qvel_dim,))

    return initial_qpos, initial_qvel

def generate_random_seq_torque(model, num_steps: int = 2000, skip_steps: int = 1, num_sin: int = 20, lim_amplitude: int = 10, lim_frequency: int = 1, lim_phase: float = 3.14):
    """
    Generate a smooth random torque sequence using sum of sinusoids.
    
    Args: 
        model: MuJoCo model
        num_steps: Number of data collection points (output length after subsampling)
        skip_steps: Number of simulation steps between data collection points.
                   Total simulation steps = num_steps * skip_steps
        num_sin: Number of sinusoids to sum for each torque dimension
        lim_amplitude: Maximum amplitude for sinusoids
        lim_frequency: Maximum frequency for sinusoids
        lim_phase: Maximum phase shift for sinusoids

    Returns:
        seq_torque: (num_sim_steps, torque_dim) torque at fine simulation resolution
                   where num_sim_steps = num_steps * skip_steps
    """
    torque_dim = model.nu
    num_sim_steps = num_steps * skip_steps

    amplitudes = np.random.uniform(low=0, high=lim_amplitude, size=(torque_dim, num_sin, 1))
    frequencies = np.random.uniform(low=0, high=lim_frequency, size=(torque_dim, num_sin, 1))
    phases = np.random.uniform(low=0, high=lim_phase, size=(torque_dim, num_sin, 1))

    # Create a steps array with shape (1, 1, num_sim_steps) and broadcast to (torque_dim, num_sin, num_sim_steps)
    # Time array at fine simulation resolution
    steps = np.arange(num_sim_steps) * model.opt.timestep  # shape (num_sim_steps,)
    steps = steps[None, None, :]  # shape (1, 1, num_sim_steps)
    steps = np.tile(steps, (torque_dim, num_sin, 1))  # shape (torque_dim, num_sin, num_sim_steps)
    
    # Use sin functions to generate seq_torque
    seq_torque = np.sum(amplitudes * np.sin(frequencies * steps + phases), axis=1).T
    return seq_torque

def generate_one_trajectory(model, initial_qpos, initial_qvel, seq_torque, num_steps: int = 2000, skip_steps: int = 1, data_dt: float = None):
    """
    Run forward dynamics simulation and collect data at specified intervals.
    
    Args:
        model: MuJoCo model (with timestep already set to dt)
        initial_qpos: Initial joint positions
        initial_qvel: Initial joint velocities
        seq_torque: Torque sequence at fine simulation resolution (num_sim_steps, torque_dim)
                   where num_sim_steps = num_steps * skip_steps
        num_steps: Number of data points to collect
        skip_steps: Number of simulation steps between data collection points
        data_dt: Data collection timestep (for computing derivatives). If None, uses model.opt.timestep * skip_steps
        
    Returns:
        seq_qpos: (num_steps, nq) collected positions
        seq_qvel: (num_steps, nv) collected velocities
        seq_qacc: (num_steps, nv) collected accelerations
        seq_mom: (num_steps, nv) collected momenta
        seq_mom_dot: (num_steps, nv) time derivative of momentum
        seq_energy: (num_steps,) total energy (kinetic + potential) at each timestep
    """
    data = mujoco.MjData(model)
    
    # Compute data_dt if not provided
    if data_dt is None:
        data_dt = model.opt.timestep * skip_steps

    # Reset data
    mujoco.mj_resetData(model, data)

    # Pass initial qpos and qvel to data
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel

    mujoco.mj_forward(model, data)

    # Pre-allocate arrays for speed (avoid list appends and conversion)
    nq, nv = model.nq, model.nv
    seq_qpos = np.empty((num_steps, nq), dtype=np.float64)
    seq_qvel = np.empty((num_steps, nv), dtype=np.float64)
    seq_qacc = np.empty((num_steps, nv), dtype=np.float64)
    seq_mom = np.empty((num_steps, nv), dtype=np.float64)
    seq_energy = np.empty((num_steps,), dtype=np.float64)  # Total energy (KE + PE)
    
    # Pre-allocate mass matrix
    M = np.zeros((nv, nv), dtype=np.float64)
    
    unstable = False
    
    # Nested loop: outer for data collection points, inner for simulation sub-steps
    # This avoids modulo check every iteration
    for data_idx in range(num_steps):
        # Run skip_steps simulation steps
        for sub_step in range(skip_steps):
            sim_idx = data_idx * skip_steps + sub_step
            data.ctrl[:] = seq_torque[sim_idx]
            mujoco.mj_step(model, data)
        
        # Check stability only at collection points (not every sim step)
        if np.any(np.isnan(data.qpos)) or np.any(np.abs(data.qvel) > 1e3):
            # Fill remaining with NaNs
            seq_qpos[data_idx:] = np.nan
            seq_qvel[data_idx:] = np.nan
            seq_qacc[data_idx:] = np.nan
            seq_mom[data_idx:] = np.nan
            seq_energy[data_idx:] = np.nan
            unstable = True
            break
        
        # Collect data (no condition check needed - always collect here)
        seq_qpos[data_idx] = data.qpos
        seq_qvel[data_idx] = data.qvel
        seq_qacc[data_idx] = data.qacc
        
        # Compute mass matrix and momentum
        mujoco.mj_fullM(model, M, data.qM)
        seq_mom[data_idx] = M @ data.qvel
        
        # Compute total energy: kinetic (data.energy[0]) + potential (data.energy[1])
        seq_energy[data_idx] = data.energy[0] + data.energy[1]

    # Compute the derivative of momentum with respect to time using central difference
    # Note: Use data_dt (collection timestep) for derivative computation
    seq_mom_dot = np.empty_like(seq_mom)
    seq_mom_dot[1:-1] = (seq_mom[2:] - seq_mom[:-2]) / (2.0 * data_dt)
    seq_mom_dot[0] = (-3*seq_mom[0] + 4*seq_mom[1] - seq_mom[2]) / (2.0 * data_dt)
    seq_mom_dot[-1] = (3*seq_mom[-1] - 4*seq_mom[-2] + seq_mom[-3]) / (2.0 * data_dt)
        
    return seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot, seq_energy

def generate(model, lim_qpos=math.pi/3, lim_qvel=math.pi/3, num_sin=5, lim_amplitude=0.5, lim_frequency=6*math.pi, lim_phase=2*math.pi, num_steps: int = 1000, skip_steps: int = 1, data_dt: float = None, use_torque=True):
    """
    Generate a single trajectory with random initial conditions and torque.
    
    Args:
        model: MuJoCo model (with timestep already set to dt)
        lim_qpos: Limit for random initial position sampling
        lim_qvel: Limit for random initial velocity sampling
        num_sin: Number of sinusoids for torque generation (default: 5 for smoother torques)
        lim_amplitude: Maximum torque amplitude (default: 0.5 for bounded energy injection)
        lim_frequency: Maximum torque frequency (default: 25 for more work cancellation)
        lim_phase: Maximum torque phase
        num_steps: Number of data points to collect
        skip_steps: Number of simulation steps between data collection points
        data_dt: Data collection timestep (for metadata). If None, uses model.opt.timestep * skip_steps
        use_torque: Whether to apply random torque or zero torque
        
    Returns:
        result: dict with seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot, seq_torque, seq_energy
               All sequences have length num_steps (data collection points)
               seq_torque is subsampled from fine resolution to match data collection
    """
    # Compute data_dt if not provided
    if data_dt is None:
        data_dt = model.opt.timestep * skip_steps
        
    # Randomly sample initial position and velocity
    initial_qpos, initial_qvel = randomly_initialize_qpos_qvel_qacc(model, lim_qpos=lim_qpos, lim_qvel=lim_qvel)
    
    # Randomly sample a smooth sequence of torque at fine simulation resolution
    num_sim_steps = num_steps * skip_steps
    if use_torque:
        seq_torque_fine = generate_random_seq_torque(
            model, num_steps=num_steps, skip_steps=skip_steps,
            num_sin=num_sin, lim_amplitude=lim_amplitude, 
            lim_frequency=lim_frequency, lim_phase=lim_phase
        )
    else:
        seq_torque_fine = np.zeros((num_sim_steps, model.nu))
    
    # Generate trajectories
    seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot, seq_energy = generate_one_trajectory(
        model, initial_qpos, initial_qvel, seq_torque_fine, 
        num_steps=num_steps, skip_steps=skip_steps, data_dt=data_dt
    )
    
    # Subsample torque to match data collection points
    # Take the torque at each data collection point (every skip_steps)
    seq_torque = seq_torque_fine[skip_steps-1::skip_steps]  # Shape: (num_steps, torque_dim)

    result = {
        'seq_qpos': seq_qpos[:],
        'seq_qvel': seq_qvel[:],
        'seq_qacc': seq_qacc[:],
        'seq_mom': seq_mom[:],
        'seq_mom_dot': seq_mom_dot[:],
        'seq_torque': seq_torque[:],
        'seq_energy': seq_energy[:]
    }
    return result

def init_plot(dim: int, title: str):
    """
    Initializes a figure with subplots for each dimension.
    Args:
        dim: Number of dimensions (rows).
        title: Title of the figure.
    Returns:
        fig: The matplotlib figure object.
        axes: Array of axes objects.
    """
    fig, axes = plt.subplots(dim, 1, figsize=(10, 10), dpi=150, sharex=True)
    if dim == 1:
        axes = np.array([axes])
    fig.suptitle(title)
    return fig, axes

def plot_on_axes(axes, seq, label: str = None):
    """
    Plots a sequence onto existing axes.
    Args:
        axes: Array of axes objects.
        seq: Sequence data of shape (num_steps, dim).
        label: Label for the plot legend.
    """
    steps = np.arange(seq.shape[0])
    dim = seq.shape[1]

    for i in range(dim):
        axes[i].scatter(steps, seq[:,i], s=1, alpha=0.6)

def generate_single_trajectory_task(args):
    """
    Worker function for multiprocessing.
    MuJoCo models cannot be pickled, so we load the model in each worker process.
    
    Args:
        args: Tuple of (xml_path, num_steps, dt, data_dt, seed, use_torque)
              - xml_path: Path to MuJoCo XML model
              - num_steps: Number of data points to collect
              - dt: Fine simulation timestep
              - data_dt: Data collection timestep (coarser)
              - seed: Random seed for this trajectory
              - use_torque: Whether to apply random torque
              
    Returns:
        Result from generate() function
    """
    xml_path, num_steps, dt, data_dt, seed, use_torque = args
    # Set the seed for this process to ensure different trajectories
    np.random.seed(seed)
    
    # Compute skip_steps from timesteps
    skip_steps = int(round(data_dt / dt))
    if skip_steps < 1:
        skip_steps = 1
    
    # Load model locally in each process (models can't be pickled)
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    # Enable energy computation for energy-based filtering
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    
    return generate(model, num_steps=num_steps, skip_steps=skip_steps, data_dt=data_dt, use_torque=use_torque)

def plot_coverage(results):
    nrows = results[0]['seq_qpos'].shape[1]
    ncols = results[0]['seq_mom'].shape[1]
    fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=(40, 20), dpi=150)

    for result in results:
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j].scatter(result['seq_qpos'][:, i], result['seq_mom'][:,j], s=0.1, alpha=0.5)
    fig.savefig('/home/gsang/Projects/Perceiver_IO/plots/coverage_forward.jpg')
    plt.close()

def plot_data(results):
    """
    Plot each seq_xxx (pos, vel, acc, momentum, momentum_dot, torque).
    Each plot shows all dimensions for all trajectories over time.
    Uses vectorized plotting for speed.
    """
    if not results:
        print("No results to plot.")
        return
    
    keys = ['seq_qpos', 'seq_qvel', 'seq_qacc', 'seq_mom', 'seq_mom_dot', 'seq_torque']
    save_dir = '/home/gsang/Projects/Perceiver_IO/plots'
    os.makedirs(save_dir, exist_ok=True)
    
    colors = plt.cm.tab10.colors  # Use a colormap for dimensions
    
    for key in keys:
        # Stack as (num_trajectories, num_steps, dim)
        arr = np.stack([result[key] for result in results], axis=0)
        num_traj, num_steps, dim = arr.shape

        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create x-coordinates: tile timesteps for all trajectories
        x = np.tile(np.arange(num_steps), num_traj)  # shape: (num_traj * num_steps,)
        
        for d in range(dim):
            # Flatten all trajectories for this dimension
            y = arr[:, :, d].ravel()  # shape: (num_traj * num_steps,)
            ax.scatter(x, y, s=1, alpha=0.3, c=[colors[d % len(colors)]], label=f'dim {d}')

        ax.set_xlabel('Timestep')
        ax.set_ylabel(key)
        ax.set_title(f"{key}: all trajectories & all dims")
        ax.legend(loc='upper right', markerscale=5)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{key}_trajectories.png"))
        plt.close(fig)

def show_statistics(results):
    """
    Concisely print statistics for the list of result dicts.
    Each dict contains arrays for seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot, seq_torque, seq_energy.
    """
    if not results:
        print("No results to compute statistics.")
        return
    
    keys = ['seq_qpos', 'seq_qvel', 'seq_qacc', 'seq_mom', 'seq_mom_dot', 'seq_torque', 'seq_energy']
    stats = {}
    for key in keys:
        data = np.concatenate([r[key] for r in results], axis=0)
        stats[key] = {
            "shape": data.shape,
            "min": np.min(data),
            "max": np.max(data),
            "mean": np.mean(data),
            "std": np.std(data),
        }
    print("---- Statistics ----")
    for key in keys:
        s = stats[key]
        print(f"{key}: shape={s['shape']}, min={s['min']:.4f}, max={s['max']:.4f}, mean={s['mean']:.4f}, std={s['std']:.4f}")
    print("--------------------")

def is_available(result, lim_energy_ratio=10.0, max_energy_limit=1000.0, 
                  max_velocity=50.0, max_acceleration=500.0) -> bool:
    """
    Check if trajectory is stable using energy and kinematic filtering.
    
    Args:
        result: dict containing arrays for 'seq_qpos', 'seq_qvel', 'seq_qacc', 'seq_energy'
        lim_energy_ratio: Maximum allowed ratio of max/initial energy
        max_energy_limit: Hard limit on maximum energy value
        max_velocity: Maximum allowed angular velocity (rad/s)
        max_acceleration: Maximum allowed angular acceleration (rad/s²)
        
    Returns:
        bool: True if trajectory is numerically stable and physically plausible, else False.
        
    Filtering criteria:
        - NaN/Inf values (numerical instability)
        - Gimbal lock avoidance for 3-hinge model
        - Energy bounds (ratio and absolute)
        - Kinematic bounds (velocity and acceleration)
    """
    seq_qpos = result['seq_qpos']    # shape: (num_steps, nq)
    seq_qvel = result['seq_qvel']    # shape: (num_steps, nv)
    seq_qacc = result['seq_qacc']    # shape: (num_steps, na)
    seq_energy = result['seq_energy']  # shape: (num_steps,)

    # Check for NaNs or Infs (numerical instability)
    if np.any(np.isnan(seq_qpos)) or np.any(np.isnan(seq_qvel)) or np.any(np.isnan(seq_qacc)):
        return False
    if np.any(np.isinf(seq_qpos)) or np.any(np.isinf(seq_qvel)) or np.any(np.isinf(seq_qacc)):
        return False
    if np.any(np.isnan(seq_energy)) or np.any(np.isinf(seq_energy)):
        return False
    
    # Kinematic limits (physically plausible dynamics)
    if np.any(np.abs(seq_qvel) > max_velocity):
        return False
    if np.any(np.abs(seq_qacc) > max_acceleration):
        return False
    
    # Check gimbal lock avoidance: qpos of second joint must be < pi/2
    if seq_qpos.shape[1] >= 2:
        if np.any(np.abs(seq_qpos[:, 1]) >= (np.pi / 2)):
            return False

    # Energy-based filtering: check that energy doesn't grow unboundedly
    initial_energy = np.abs(seq_energy[0]) + 1.0  # Add 1.0 to avoid division by zero
    max_energy = np.max(np.abs(seq_energy))
    energy_ratio = max_energy / initial_energy
    
    if energy_ratio > lim_energy_ratio:
        return False
    
    # Hard limit on maximum energy
    if max_energy > max_energy_limit:
        return False

    return True

def save_as_h5py(results, save_path, xml_path, num_steps, num_trajectories, dt, data_dt):
    """
    Save trajectory results to HDF5 file.
    
    Args:
        results: List of trajectory result dicts
        save_path: Directory to save the file
        xml_path: Path to MuJoCo XML model
        num_steps: Number of data points per trajectory
        num_trajectories: Number of trajectories
        dt: Fine simulation timestep
        data_dt: Data collection timestep
    """
    # Create directory if it doesn't exist (including parent directories)
    # Don't use os.mkdir(), cause it only creates one level
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'traj_{num_trajectories}-steps_{num_steps}.h5') 

    # plot_coverage(results)
    # plot_data(results)
    show_statistics(results)

    # Don't forget to set the write mode   
    with h5py.File(file_path, 'w') as file:
        with open(xml_path, 'r') as xml_file:
            file.attrs['xml'] = xml_file.read()
        file.attrs['num_steps'] = num_steps
        file.attrs['num_trajectories'] = num_trajectories
        file.attrs['dt'] = dt              # Fine simulation timestep
        file.attrs['data_dt'] = data_dt    # Data collection timestep
        file.attrs['skip_steps'] = int(round(data_dt / dt))  # Number of sim steps between data points

        for i, result in tqdm(enumerate(results), total=len(results), desc='Saving'):
            group = file.create_group(f'traj_{i}')
            group.create_dataset('seq_qpos', data=result['seq_qpos'], dtype='f4')
            group.create_dataset('seq_qvel', data=result['seq_qvel'], dtype='f4')
            group.create_dataset('seq_qacc', data=result['seq_qacc'], dtype='f4')
            group.create_dataset('seq_mom', data=result['seq_mom'], dtype='f4')
            group.create_dataset('seq_mom_dot', data=result['seq_mom_dot'], dtype='f4')
            group.create_dataset('seq_torque', data=result['seq_torque'], dtype='f4')
            group.create_dataset('seq_energy', data=result['seq_energy'], dtype='f4')


def main():
    # -----------------------------------------
    # -------------Configurations--------------
    #------------------------------------------
    save_path = '/home/gsang/Projects/Perceiver_IO/data'
    xml_path = '/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml'
    num_steps = 4000          # Number of data points to collect per trajectory (reduced for stability)
    num_trajectories = 40000   # Target number of available trajectories (increased for total data volume)
    dt = 0.0001               # Fine simulation timestep (physics accuracy)
    data_dt = 0.00025         # Data collection timestep (coarser, for dataset)
    batch_size = 1000         # Generate in batches for efficiency
    
    # Compute skip_steps for display
    skip_steps = int(round(data_dt / dt))
    print(f"Configuration:")
    print(f"  Simulation timestep (dt): {dt}")
    print(f"  Data collection timestep (data_dt): {data_dt}")
    print(f"  Skip steps: {skip_steps} (collect every {skip_steps} simulation steps)")
    print(f"  Total simulation steps per trajectory: {num_steps * skip_steps}")
    print(f"  Data points per trajectory: {num_steps}")
    
    # Filter limits
    lim_energy_ratio = 5.0  # Maximum allowed ratio of max/initial energy
    max_energy_limit = 500.0  # Hard limit on maximum energy
    max_velocity = 10.0  # Maximum angular velocity (rad/s)
    max_acceleration = 100.0  # Maximum angular acceleration (rad/s²)

    use_torque = True
    available_results = []
    total_generated = 0
    seed_counter = 0
    
    pbar = tqdm(total=num_trajectories, desc="Collecting available trajectories")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        while len(available_results) < num_trajectories:
            # Generate batch of seeds
            seeds = np.random.randint(0, 99999, (batch_size,))
            seed_counter += batch_size
            
            # Pass both dt and data_dt to worker tasks
            tasks = [(xml_path, num_steps, dt, data_dt, seed, use_torque) for seed in seeds]
            results = list(executor.map(generate_single_trajectory_task, tasks))
            total_generated += len(results)
            
            # Filter and accumulate (energy + kinematic filtering)
            new_available = [r for r in results if is_available(r, lim_energy_ratio, max_energy_limit, max_velocity, max_acceleration)]
            available_results.extend(new_available)
            
            # Update progress bar
            pbar.n = min(len(available_results), num_trajectories)
            pbar.set_postfix(generated=total_generated, accept_rate=f"{len(available_results)/total_generated:.1%}")
            pbar.refresh()
    
    pbar.close()
    
    # Trim to exact number requested
    available_results = available_results[:num_trajectories]
    print(f"Collected {len(available_results)} trajectories (generated {total_generated}, accept rate: {len(available_results)/total_generated:.1%})")
    
    save_as_h5py(available_results, save_path, xml_path, num_steps, num_trajectories, dt, data_dt)
    print(f"All trajectories are saved as a h5py file to {save_path}")
    
if __name__ == '__main__':
    main()