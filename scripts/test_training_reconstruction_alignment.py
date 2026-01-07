"""
Verify whether training trajectories align with MuJoCo reconstruction.

Important nuance:
- The dataset stores seq_torque at *data_dt* resolution.
- The original dataset generation used a *fine* torque sequence at dt resolution and then
  sub-sampled torque for storage (see generate_dataset_forward.py).
- The reconstruction utilities in src/models/utils.py assume the stored torque is held
  constant for skip_steps simulation steps.

So this test answers:
  "Do training trajectories align with reconstruction under the *stored-torque* assumption?"
Not:
  "Can we perfectly reproduce the generator trajectories?" (we can't without the fine torque).
"""

import sys
from pathlib import Path
import tempfile

import numpy as np
import torch


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def main():
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from scripts.dataset import TrajectoryDPFCached
    from src.models.utils import reconstruct_traj_with_momentum
    import mujoco

    data_path = "/home/gsang/Projects/Perceiver_IO/data/traj_4000-steps_4000.h5"
    trajectory_length = 1000
    num_test = 25

    dataset = TrajectoryDPFCached(data_path, trajectory_length=trajectory_length)
    dt = float(dataset.dt)
    data_dt = float(dataset.data_dt)
    xml = dataset.xml

    if xml is None:
        raise RuntimeError("Dataset does not contain MuJoCo xml (h5 attr 'xml').")

    # Write XML to temp file for mujoco loader
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml)
        xml_path = f.name

    model = mujoco.MjModel.from_xml_path(xml_path)

    print(f"Dataset: {data_path}")
    print(f"Trajectories: {len(dataset)}, trajectory_length={trajectory_length}")
    print(f"dt={dt}, data_dt={data_dt}, skip_steps={int(round(data_dt/dt))}")
    print()

    mse_q_list = []
    mse_p_list = []

    for i in range(min(num_test, len(dataset))):
        sample = dataset[i]
        q = sample["seq_qpos"].numpy()  # [T, nq]
        p = sample["seq_mom"].numpy()   # [T, nv]
        tau = sample["seq_torque"].numpy()  # [T, nu]

        data = mujoco.MjData(model)
        data.qpos[:] = q[0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        M = np.zeros((model.nv, model.nv), dtype=np.float64)
        mujoco.mj_fullM(model, M, data.qM)
        qvel0 = np.linalg.solve(M, p[0])

        recon = reconstruct_traj_with_momentum(
            model,
            num_steps=len(q),
            dt=dt,
            initial_qpos=q[0],
            initial_qvel=qvel0,
            seq_torque=tau,
            data_dt=data_dt,
        )

        # reconstruct_traj_with_momentum returns length (T-1) for qpos/mom, torque[:-1]
        q_recon = recon["seq_qpos"]      # [T-1, nq]
        p_recon = recon["seq_mom"]       # [T-1, nv]

        # Align with dataset: compare dataset[1:] to recon
        q_tgt = q[1:]
        p_tgt = p[1:]

        mq = mse(q_tgt, q_recon)
        mp = mse(p_tgt, p_recon)

        mse_q_list.append(mq)
        mse_p_list.append(mp)

        if i < 5:
            print(f"traj {i:3d}: MSE(qpos)={mq:.6e}, MSE(mom)={mp:.6e}")

    mse_q_arr = np.array(mse_q_list)
    mse_p_arr = np.array(mse_p_list)

    print("\nSummary (stored-torque reconstruction assumption):")
    print(f"  MSE(qpos): mean={mse_q_arr.mean():.6e}, std={mse_q_arr.std():.6e}, min={mse_q_arr.min():.6e}, max={mse_q_arr.max():.6e}")
    print(f"  MSE(mom):  mean={mse_p_arr.mean():.6e}, std={mse_p_arr.std():.6e}, min={mse_p_arr.min():.6e}, max={mse_p_arr.max():.6e}")

    print("\nInterpretation:")
    print("- If these MSEs are NOT near ~0, that's expected: the dataset was generated using a *fine* torque sequence,")
    print("  but only a sub-sampled torque is stored, so reconstruction is under-determined.")
    print("- This directly explains why a 'low HNN energy' trajectory may still not align with MuJoCo reconstruction.")


if __name__ == '__main__':
    main()






