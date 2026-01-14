# -*- coding: utf-8 -*-
import mujoco
import numpy as np
import torch
import tqdm
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import itertools
# -------------------------------------------------------------------
# 1. MuJoCo model string
# -------------------------------------------------------------------
xml_path = "/home/gsang/Projects/Perceiver_IO/configs/rigid_arm_hinge.xml"

# -------------------------------------------------------------------
# 2. Planners & Helpers
# -------------------------------------------------------------------
def quintic_planning(p_s, v_s, a_s, p_e, v_e, a_e, t):
    c0 = p_s
    c1 = v_s
    c2 = a_s / 2.0
    delta_p = p_e - p_s
    delta_v = v_e - v_s
    delta_a = a_e - a_s
    T = t[-1] - t[0]
    c3 = (delta_a * T**2 + 20 * delta_p - 8 * delta_v * T) / (2 * T**3)
    c4 = (-delta_a * T**2 - 15 * delta_p + 7 * delta_v * T) / (T**4)
    c5 = (delta_a * T**2 + 12 * delta_p - 6 * delta_v * T) / (2 * T**5)
    
    p_t = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    v_t = c1 + 2 * c2 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
    a_t = 2 * c2 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
    return p_t.T, v_t.T, a_t.T  # Transpose immediately to (T, D)

def get_t(num_timesteps, dt):
    return np.array([i * dt for i in range(num_timesteps)])

# -------------------------------------------------------------------
# 3. WORKER FUNCTION (Optimized)
# -------------------------------------------------------------------
def generate_small_batch(args):
    """
    Generates a small batch (e.g., 50 trajs). 
    Smaller functions prevent 'hanging' UI.
    """
    batch_size, num_timesteps, dt, seed_offset = args
    
    # Local Initialization
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt
    dim = model.nv
    
    rng = np.random.RandomState(seed_offset)
    t = get_t(num_timesteps, dt)
    
    # Pre-allocation for speed
    q_list, v_list, a_list, tau_list, p_list, dp_list = [], [], [], [], [], []
    
    M = np.zeros((dim, dim))

    for _ in range(batch_size):  
        traj_tau = np.zeros((num_timesteps, dim))
        traj_mom = np.zeros((num_timesteps, dim))
        
        # UNIFORM SAMPLING (matching generate_no_torque_dataset.py logic)
        # q in [-pi/4, pi/4], v in [-1, 1] roughly
        q_s = rng.uniform(low=-np.pi/4, high=np.pi/4, size=(dim, 1))
        q_e = rng.uniform(low=-np.pi/4, high=np.pi/4, size=(dim, 1))
        
        # Velocities: MJP implies velocities. We can set start/end velocities.
        qd_s = rng.uniform(low=-1.0, high=1.0, size=(dim, 1))
        qd_e = rng.uniform(low=-1.0, high=1.0, size=(dim, 1))
        
        qdd_s = np.zeros((dim, 1)) # Zero accel start
        qdd_e = np.zeros((dim, 1)) # Zero accel end

        q_t, v_t, a_t = quintic_planning(q_s, qd_s, qdd_s, q_e, qd_e, qdd_e, t)

        mujoco.mj_resetData(model, data)

        for k in range(num_timesteps):
            # Update State
            data.qpos[:] = q_t[k]
            data.qvel[:] = v_t[k]
            data.qacc[:] = a_t[k]
            
            # Inverse Dynamics
            mujoco.mj_inverse(model, data)
            traj_tau[k] = data.qfrc_inverse

            # Momentum (Forward Kinematics for Mass Matrix)
            mujoco.mj_forward(model, data)
            mujoco.mj_fullM(model, M, data.qM)
            traj_mom[k] = M @ v_t[k]

        # --- Derivatives ---
        traj_mom_dot = np.zeros_like(traj_mom)
        traj_mom_dot[1:-1] = (traj_mom[2:] - traj_mom[:-2]) / (2.0 * dt)
        traj_mom_dot[0] = (-3*traj_mom[0] + 4*traj_mom[1] - traj_mom[2]) / (2.0*dt)
        traj_mom_dot[-1] = (3*traj_mom[-1] - 4*traj_mom[-2] + traj_mom[-3]) / (2.0*dt)

        # Collect
        q_list.append(q_t)
        v_list.append(v_t)
        a_list.append(a_t)
        tau_list.append(traj_tau)
        p_list.append(traj_mom)
        dp_list.append(traj_mom_dot)

    return q_list, v_list, a_list, tau_list, p_list, dp_list

def plot_trajectory(q_t, v_t, a_t, momentum_t, momentum_dot_t, torque_t, type):
    num_subplots = q_t.shape[1] + v_t.shape[1] + a_t.shape[1] + momentum_t.shape[1] + momentum_dot_t.shape[1]
    if torque_t is not None:
        num_subplots += torque_t.shape[1]
        num_rows = 6
    else:
        num_rows = 5
    
    num_cols = num_subplots // num_rows
    plt.figure(figsize=(20*num_cols, 10*num_rows), dpi=300)
    # set font size to 24
    plt.rcParams.update({'font.size': 24})
    plt.rcParams.update({'legend.fontsize': 24})
    plt.rcParams.update({'figure.titlesize': 24})
    plt.rcParams.update({'figure.titleweight': 'bold'})
    plt.rcParams.update({'axes.labelsize': 24})
    # set linewidth to 4
    plt.rcParams.update({'lines.linewidth': 4})
    for i in range(q_t.shape[1]):
        plt.subplot(num_rows, num_cols, i+1)
        plt.plot(q_t[:, i], label=f'q_{i}')
        plt.legend()
    for i in range(v_t.shape[1]):
        plt.subplot(num_rows, num_cols, i+1+q_t.shape[1])
        plt.plot(v_t[:, i], label=f'v_{i}')
        plt.legend()
    for i in range(a_t.shape[1]):
        plt.subplot(num_rows, num_cols, i+1+q_t.shape[1]+v_t.shape[1])
        plt.plot(a_t[:, i], label=f'a_{i}')
        plt.legend()
    for i in range(momentum_t.shape[1]):
        plt.subplot(num_rows, num_cols, i+1+q_t.shape[1]+v_t.shape[1]+a_t.shape[1])
        plt.plot(momentum_t[:, i], label=f'momentum_{i}')
        plt.legend()
    for i in range(momentum_dot_t.shape[1]):
        plt.subplot(num_rows, num_cols, i+1+q_t.shape[1]+v_t.shape[1]+a_t.shape[1]+momentum_t.shape[1])
        plt.plot(momentum_dot_t[:, i], label=f'momentum_dot_{i}')
        plt.legend()
    if torque_t is not None:
        for i in range(torque_t.shape[1]):
            plt.subplot(num_rows, num_cols, i+1+q_t.shape[1]+v_t.shape[1]+a_t.shape[1]+momentum_t.shape[1]+momentum_dot_t.shape[1])
            plt.plot(torque_t[:, i], label=f'torque_{i}')
            plt.legend()
    plt.savefig(f'/home/gsang/Projects/Perceiver_IO/data/{type}.png')   
    plt.close()


def plot_coverage(all_q, all_p, save_path='coverage.png'):
    """
    all_q: (N, dim)
    all_p: (N, dim)
    Creates C(2*dim, 2) scatter plots to verify coverage.
    """
    dim = all_q.shape[1]

    labels = [f'q_{i}' for i in range(dim)] + [f'p_{i}' for i in range(dim)]
    data_matrix = np.hstack([all_q, all_p])  # (N, 2*dim)

    num_vars = 2 * dim
    pairs = list(itertools.combinations(range(num_vars), 2))
    num_plots = len(pairs)

    cols = 3
    rows = (num_plots + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 4 * rows))

    N = data_matrix.shape[0]
    indices = np.random.choice(N, size=min(N, 10000), replace=False)

    for i, (idx1, idx2) in enumerate(pairs):
        plt.subplot(rows, cols, i + 1)
        x = data_matrix[indices, idx1]
        y = data_matrix[indices, idx2]

        plt.scatter(x, y, s=1, alpha=0.5)
        plt.xlabel(labels[idx1])
        plt.ylabel(labels[idx2])
        plt.title(f'{labels[idx1]} vs {labels[idx2]}')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Coverage plot saved to {save_path}")
    plt.close()



# -------------------------------------------------------------------
# 4. Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Safer multiprocessing for libraries like MuJoCo
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- Config ---
    TOTAL_TRAJ = 1000
    CHUNK_SIZE = 1   # Small chunk size updates progress bar frequently
    NUM_TIMESTEPS = 2000
    DT = 0.001
    
    num_chunks = TOTAL_TRAJ // CHUNK_SIZE
    num_workers = os.cpu_count() - 16
    
    print(f"Generating {TOTAL_TRAJ} trajectories")
    print(f"Workers: {num_workers} | Chunk Size: {CHUNK_SIZE} | Total Chunks: {num_chunks}")

    # Prepare tasks
    tasks = []
    for i in range(num_chunks):
        # Unique seed per chunk
        seed = np.random.randint(0, 1000000) + i
        tasks.append((CHUNK_SIZE, NUM_TIMESTEPS, DT, seed))

    # Storage
    data_store = {k: [] for k in ['q_t', 'v_t', 'a_t', 'torque', 'momentum_t', 'momentum_dot_t']}

    # Parallel execution
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to track completed chunks
        results = list(tqdm.tqdm(
            executor.map(generate_small_batch, tasks), 
            total=num_chunks,
            unit="chunk"
        ))

    print("Aggregating results...")
    for res in tqdm.tqdm(results, total=len(results), desc="Aggregating results"):
        q, v, a, tau, p, dp = res
        data_store['q_t'].extend(q)
        data_store['v_t'].extend(v)
        data_store['a_t'].extend(a)
        data_store['torque'].extend(tau)
        data_store['momentum_t'].extend(p)
        data_store['momentum_dot_t'].extend(dp)
    print("---------------------------------------------------")
    print(f"q range: {np.array(data_store['q_t']).min()}, {np.array(data_store['q_t']).max()}")
    print(f"p range: {np.array(data_store['momentum_t']).min()}, {np.array(data_store['momentum_t']).max()}")
    print(f"v range: {np.array(data_store['v_t']).min()}, {np.array(data_store['v_t']).max()}")
    print(f"dp range: {np.array(data_store['momentum_dot_t']).min()}, {np.array(data_store['momentum_dot_t']).max()}")
    print(f"tau range: {np.array(data_store['torque']).min()}, {np.array(data_store['torque']).max()}")
    print(f"q mean: {np.array(data_store['q_t']).mean()}, q std: {np.array(data_store['q_t']).std()}")
    print(f"p mean: {np.array(data_store['momentum_t']).mean()}, p std: {np.array(data_store['momentum_t']).std()}")
    print(f"v mean: {np.array(data_store['v_t']).mean()}, v std: {np.array(data_store['v_t']).std()}")
    print(f"dp mean: {np.array(data_store['momentum_dot_t']).mean()}, dp std: {np.array(data_store['momentum_dot_t']).std()}")
    print(f"tau mean: {np.array(data_store['torque']).mean()}, tau std: {np.array(data_store['torque']).std()}")
    print("---------------------------------------------------")

    print("Converting to Tensors...")
    final_data = {
        k: torch.tensor(np.array(v), dtype=torch.float32) 
        for k, v in data_store.items()
    }
    
    # Coverage visualization (flatten to (N*T, dim))
    flat_q = final_data['q_t'].reshape(-1, final_data['q_t'].shape[-1]).numpy()
    flat_p = final_data['momentum_t'].reshape(-1, final_data['momentum_t'].shape[-1]).numpy()
    os.makedirs('/home/gsang/Projects/Perceiver_IO/data', exist_ok=True)
    plot_coverage(flat_q, flat_p, save_path='/home/gsang/Projects/Perceiver_IO/data/coverage_3dof_hinge_forced.png')

    # Save
    save_path = f'/home/gsang/Projects/Perceiver_IO/data/trajectory_3_dof_hinge_with_torque_{TOTAL_TRAJ*NUM_TIMESTEPS}.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(final_data, save_path)
    print(f"Saved to {save_path}")