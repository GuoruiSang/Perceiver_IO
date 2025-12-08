import mujoco
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import h5py
import os
from tqdm import tqdm

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

def generate_random_seq_torque(model, num_steps: int = 2000, num_sin: int = 20, lim_amplitude: int = 10, lim_frequency: int = 1, lim_phase: float = 3.14):
    """
        Args: 
            model:
            num_steps:
            num_sin:
            lim_amplitude:
            lim_frequency:
            lim_phase:

        Returns:

    """
    torque_dim = model.nu

    amplitudes = np.random.uniform(low=0, high=lim_amplitude, size=(torque_dim, num_sin, 1))
    frequencies = np.random.uniform(low=0, high=lim_frequency, size=(torque_dim, num_sin, 1))
    phases = np.random.uniform(low=0, high=lim_phase, size=(torque_dim, num_sin, 1))

    # Create a steps array with shape (1, 1, num_steps) and broadcast to (torque_dim, num_sin, num_steps)
    steps = np.arange(num_steps) * model.opt.timestep  # shape (num_steps,)
    steps = steps[None, None, :]  # shape (1, 1, num_steps)
    steps = np.tile(steps, (torque_dim, num_sin, 1))  # shape (torque_dim, num_sin, num_steps)
    
    # Use sin functions to generate seq_torque
    seq_torque = np.sum(amplitudes * np.sin(frequencies * steps + phases), axis=1).T
    # seq_torque = np.random.uniform(-10, 10, (num_steps, torque_dim))
    # seq_torque = 0.1 * np.ones((num_steps, torque_dim))
    return seq_torque

def generate_one_trajectory(model, initial_qpos, initial_qvel, seq_torque, num_steps: int = 2000, trim_length=10):
    """
        Args:
            model
            initial_qpos
            initial_qvel
            seq_torque
            num_steps
        Returns:
            seq_qpos
            seq_qvel
            seq_qacc
            seq_mom
            seq_mom_dot
    """
    data = mujoco.MjData(model)

    # Reset data
    mujoco.mj_resetData(model, data)

    # Pass initial qpos and qvel to data
    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel

    mujoco.mj_forward(model, data)
    # Record the initial qpos, qvel, qacc, and momentum
    # seq_qpos = [data.qpos]
    # seq_qvel = [data.qvel]
    # seq_qacc = [data.qacc]

    seq_qpos = []
    seq_qvel = []
    seq_qacc = []

    # Get initial mass matrix
    M = np.zeros((model.nv, model.nv))
    # mujoco.mj_fullM(model, M, data.qM)
    # seq_mom = [M @ data.qvel]
    seq_mom = []

    # Run forward dynamics. 
    # State 0 -> torque 0 -> State 1, ..., so if the length of seq_torque is N, then the number of states we'll record is N+1
    for i in range(num_steps):
        # Send a torque signal to the actuator
        data.ctrl = seq_torque[i]
        
        # Run one step of dynamics
        # Don't use mujoco.mj_forward, cause it only infer other related values, but doesn't integrate over the time
        mujoco.mj_step(model, data)

        # Record the states after applying control
        seq_qpos.append(data.qpos.copy())
        seq_qvel.append(data.qvel.copy())
        seq_qacc.append(data.qacc.copy())

        # Recompute the mass matrix, cause it changes
        mujoco.mj_fullM(model, M, data.qM)
        # Record the momentum
        seq_mom.append(M @ data.qvel)


    # Convert lists to np.array before returning
    seq_qpos = np.array(seq_qpos)
    seq_qvel = np.array(seq_qvel)
    seq_qacc = np.array(seq_qacc)
    seq_mom = np.array(seq_mom)

    # Compute the derivative of momentum with respect to time using central difference
    seq_mom_dot = np.zeros_like(seq_mom)
    seq_mom_dot[1:-1] = (seq_mom[2:] - seq_mom[:-2]) / (2.0 * model.opt.timestep)
    seq_mom_dot[0] = (-3*seq_mom[0] + 4*seq_mom[1] - seq_mom[2]) / (2.0*model.opt.timestep)
    seq_mom_dot[-1] = (3*seq_mom[-1] - 4*seq_mom[-2] + seq_mom[-3]) / (2.0*model.opt.timestep)
        
    return seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot

def generate(model, lim_qpos=0.5, lim_qvel=0.5, num_sin=1, lim_amplitude=10, lim_frequency=1, lim_phase=100, num_steps: int = 1000, use_torque=True):
    # Randomly sample initial position and velocity
    initial_qpos, initial_qvel = randomly_initialize_qpos_qvel_qacc(model, lim_qpos=lim_qpos, lim_qvel=lim_qvel)
    # Randomly sample a smooth sequence of torque
    if use_torque:
        seq_torque = generate_random_seq_torque(model, num_steps=num_steps, num_sin=num_sin, lim_amplitude=lim_amplitude, lim_frequency=lim_frequency, lim_phase=lim_phase)
    else:
        seq_torque = np.zeros((num_steps,model.nu))
    # Generate trajectories
    seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot = generate_one_trajectory(
        model, initial_qpos, initial_qvel, seq_torque, num_steps
    )

    result = {
        'seq_qpos': seq_qpos[:],
        'seq_qvel': seq_qvel[:],
        'seq_qacc': seq_qacc[:],
        'seq_mom': seq_mom[:],
        'seq_mom_dot': seq_mom_dot[:],
        'seq_torque': seq_torque[:]
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
        args: Tuple of (xml_path, num_steps, seed)
    Returns:
        Result from generate() function
    """
    xml_path, num_steps, dt, seed, use_torque = args
    # Set the seed for this process to ensure different trajectories
    np.random.seed(seed)
    
    # Load model locally in each process (models can't be pickled)
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    
    return generate(model, num_steps=num_steps, use_torque=use_torque)

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
    Each dict contains arrays for seq_qpos, seq_qvel, seq_qacc, seq_mom, seq_mom_dot, seq_torque.
    """
    if not results:
        print("No results to compute statistics.")
        return
    
    keys = ['seq_qpos', 'seq_qvel', 'seq_qacc', 'seq_mom', 'seq_mom_dot', 'seq_torque']
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

def is_available(result, lim_acc, lim_vel, lim_pos) -> bool:
    """
    Check if all timesteps of trajectory satisfy the provided limits.
    Args:
        result: dict containing arrays for 'seq_qpos', 'seq_qvel', 'seq_qacc'
        lim_acc: float/array, limit for acceleration (inclusive)
        lim_vel: float/array, limit for velocity (inclusive)
        lim_pos: float/array, limit for position (inclusive)
    Returns:
        bool: True if trajectory stays within specified limits, else False.
        Additional default rule: The qpos of the second joint must be strictly less than pi/2 at all timesteps.
    """
    seq_qpos = result['seq_qpos']    # shape: (num_steps, nq)
    seq_qvel = result['seq_qvel']    # shape: (num_steps, nv)
    seq_qacc = result['seq_qacc']    # shape: (num_steps, na)

    # Check abs(qacc) <= lim_acc everywhere
    if np.any(np.abs(seq_qacc) > lim_acc):
        return False
    # Check abs(qvel) <= lim_vel everywhere
    if np.any(np.abs(seq_qvel) > lim_vel):
        return False
    # Check abs(qpos) <= lim_pos everywhere
    if np.any(np.abs(seq_qpos) > lim_pos):
        return False
    # By default, check if the qpos of the second joint (index 1) is strictly less than pi/2 at all timesteps
    if seq_qpos.shape[1] >= 2:  # Only if there is a second joint
        if np.any(seq_qpos[:, 1] >= (np.pi / 2)):
            return False

    return True

def save_as_h5py(results, save_path, xml_path, num_steps, num_trajectories, dt):
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
        file.attrs['dt'] = dt

        for i, result in tqdm(enumerate(results), total=len(results), desc='Saving'):
            group = file.create_group(f'traj_{i}')
            group.create_dataset('seq_qpos', data=result['seq_qpos'], dtype='f4')
            group.create_dataset('seq_qvel', data=result['seq_qvel'], dtype='f4')
            group.create_dataset('seq_qacc', data=result['seq_qacc'], dtype='f4')
            group.create_dataset('seq_mom', data=result['seq_mom'], dtype='f4')
            group.create_dataset('seq_mom_dot', data=result['seq_mom_dot'], dtype='f4')
            group.create_dataset('seq_torque', data=result['seq_torque'], dtype='f4')


def main():
    # -----------------------------------------
    # -------------Configurations--------------
    #------------------------------------------
    save_path = '/home/gsang/Projects/Perceiver_IO/data'
    xml_path = '/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml'
    num_steps = 2000
    num_trajectories = 40000  # Target number of available trajectories
    dt = 0.0005
    batch_size = 1000  # Generate in batches for efficiency
    
    # Filter limits
    lim_acc, lim_vel, lim_pos = 50, 10, np.pi

    use_torque = True
    available_results = []
    total_generated = 0
    seed_counter = 0
    
    pbar = tqdm(total=num_trajectories, desc="Collecting available trajectories")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        while len(available_results) < num_trajectories:
            # Generate batch of seeds
            # seeds = np.arange(seed_counter, seed_counter + batch_size)
            seeds = np.random.randint(0, 99999, (batch_size,))
            seed_counter += batch_size
            
            tasks = [(xml_path, num_steps, dt, seed, use_torque) for seed in seeds]
            results = list(executor.map(generate_single_trajectory_task, tasks))
            total_generated += len(results)
            
            # Filter and accumulate
            new_available = [r for r in results if is_available(r, lim_acc, lim_vel, lim_pos)]
            available_results.extend(new_available)
            
            # Update progress bar
            pbar.n = min(len(available_results), num_trajectories)
            pbar.set_postfix(generated=total_generated, accept_rate=f"{len(available_results)/total_generated:.1%}")
            pbar.refresh()
    
    pbar.close()
    
    # Trim to exact number requested
    available_results = available_results[:num_trajectories]
    print(f"Collected {len(available_results)} trajectories (generated {total_generated}, accept rate: {len(available_results)/total_generated:.1%})")
    
    save_as_h5py(available_results, save_path, xml_path, num_steps, num_trajectories, dt)
    print(f"All trajectories are saved as a h5py file to {save_path}")
    
if __name__ == '__main__':
    main()