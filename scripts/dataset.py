from torch.utils.data import Dataset
import h5py
import torch
import numpy as np

class TrajectoryDPF(Dataset):
    def __init__(self, h5_path: str):
        super().__init__()
        self.h5_path = h5_path
        self.metadata = {}
        
        # Open file and keep it open for the lifetime of the dataset
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Load metadata
        for k, v in self.h5_file.attrs.items():
            self.metadata[k] = v

    def __getitem__(self, index):
        traj_data = self.h5_file[f'traj_{index}']

        item = {
            'seq_qpos': torch.from_numpy(traj_data['seq_qpos'][:]),  # [:] to load into memory
            'seq_qvel': torch.from_numpy(traj_data['seq_qvel'][:]),
            'seq_qacc': torch.from_numpy(traj_data['seq_qacc'][:]),
            'seq_mom': torch.from_numpy(traj_data['seq_mom'][:]),
            'seq_mom_dot': torch.from_numpy(traj_data['seq_mom_dot'][:]),
            'seq_torque': torch.from_numpy(traj_data['seq_torque'][:])
        }
        
        return item
    
    def __len__(self):
        return self.metadata['num_trajectories']
    
    def __del__(self):
        """Close the HDF5 file when the dataset is destroyed."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except (AttributeError, TypeError, ValueError):
                # File may already be closed or in an invalid state during garbage collection
                # This is safe to ignore as Python will clean up resources
                pass


class TrajectoryHNN(TrajectoryDPF):
    """Original HDF5-based dataset - reads from disk on each access (SLOW)."""
    def __init__(self, *args, **keywords) -> None:
        super().__init__(*args, **keywords)

    def __len__(self):
        return self.metadata['num_trajectories'] * self.metadata['num_steps']

    def __getitem__(self, index):
        idx_traj = index // self.metadata['num_steps']
        idx_data = index % self.metadata['num_steps']  

        qpos = self.h5_file[f'traj_{idx_traj}/seq_qpos'][idx_data]
        qvel = self.h5_file[f'traj_{idx_traj}/seq_qvel'][idx_data]
        qacc = self.h5_file[f'traj_{idx_traj}/seq_qacc'][idx_data]
        mom = self.h5_file[f'traj_{idx_traj}/seq_mom'][idx_data]
        mom_dot = self.h5_file[f'traj_{idx_traj}/seq_mom_dot'][idx_data]
        torque = self.h5_file[f'traj_{idx_traj}/seq_torque'][idx_data]

        item = {
            'qpos': torch.from_numpy(qpos),
            'qvel': torch.from_numpy(qvel),
            'qacc': torch.from_numpy(qacc),
            'mom': torch.from_numpy(mom),
            'mom_dot': torch.from_numpy(mom_dot),
            'torque': torch.from_numpy(torque)
        }
        return item


class TrajectoryHNNCached(Dataset):
    """Memory-cached dataset - loads ALL data into RAM once (FAST).
    
    Use this for datasets that fit in memory. Much faster than HDF5 random access.
    """
    def __init__(self, h5_path: str, trajectory_length: int = 500):
        super().__init__()
        print(f"Loading dataset into memory from {h5_path}...")
        
        with h5py.File(h5_path, 'r') as f:
            self.num_traj = f.attrs['num_trajectories']
            self.num_steps = f.attrs['num_steps']
            self.model_xml = f.attrs['xml']
            self.dt = f.attrs['dt']
            
            # Pre-allocate and load all data at once
            all_qpos, all_qvel, all_qacc = [], [], []
            all_mom, all_mom_dot, all_torque = [], [], []
            
            for i in range(self.num_traj):
                traj = f[f'traj_{i}']
                all_qpos.append(traj['seq_qpos'][:])
                all_qvel.append(traj['seq_qvel'][:])
                all_qacc.append(traj['seq_qacc'][:])
                all_mom.append(traj['seq_mom'][:])
                all_mom_dot.append(traj['seq_mom_dot'][:])
                all_torque.append(traj['seq_torque'][:])
            
            # Stack and flatten: (num_traj, num_steps, dim) -> (num_traj * num_steps, dim)
            
            self.qpos = torch.from_numpy(np.concatenate(all_qpos, axis=0))
            self.qvel = torch.from_numpy(np.concatenate(all_qvel, axis=0))
            self.qacc = torch.from_numpy(np.concatenate(all_qacc, axis=0))
            self.mom = torch.from_numpy(np.concatenate(all_mom, axis=0))
            self.mom_dot = torch.from_numpy(np.concatenate(all_mom_dot, axis=0))
            self.torque = torch.from_numpy(np.concatenate(all_torque, axis=0))
        
        print(f"Loaded {len(self)} samples into memory.")
        print("---------------Statistics--------------")
        print(f"Range of qpos: [{self.qpos.max()} - {self.qpos.min()}]; Std of qpos: {self.qpos.std()}")
        print(f"Range of qvel: [{self.qvel.max()} - {self.qvel.min()}]; Std of qvel: {self.qvel.std()}")
        print(f"Range of qacc: [{self.qacc.max()} - {self.qacc.min()}]; Std of qacc: {self.qacc.std()}")
        print(f"Range of mom: [{self.mom.max()} - {self.mom.min()}]; Std of mom: {self.mom.std()}")
        print(f"Range of mom_dot: [{self.mom_dot.max()} - {self.mom_dot.min()}]; Std of mom_dot: {self.mom_dot.std()}")
        print(f"Range of torque: [{self.torque.max()} - {self.torque.min()}]; Std of torque: {self.torque.std()}")

    
    def __len__(self):
        return len(self.qpos)
    
    def __getitem__(self, index):
        return {
            'qpos': self.qpos[index],
            'qvel': self.qvel[index],
            'qacc': self.qacc[index],
            'mom': self.mom[index],
            'mom_dot': self.mom_dot[index],
            'torque': self.torque[index]
        }

class TrajectoryDPFCached(Dataset):
    def __init__(self, h5_path: str, trajectory_length: int = 500) -> None:
        super().__init__()

        with h5py.File(h5_path, 'r') as f:
            self.num_traj = f.attrs['num_trajectories']
            self.num_steps = f.attrs['num_steps']
            
            # Load simulation metadata (with defaults for backwards compatibility)
            self.dt = f.attrs.get('dt', 0.0001)
            self.data_dt = f.attrs.get('data_dt', 0.0002)
            self.xml = f.attrs.get('xml', None)

            # Pre-allocate and load all data at once
            self.all_seq_qpos, self.all_seq_mom, self.all_seq_torque = [], [], []

            for i in range(self.num_traj):
                traj = f[f'traj_{i}']
                self.all_seq_qpos.append(traj['seq_qpos'][:trajectory_length])
                self.all_seq_torque.append(traj['seq_torque'][:trajectory_length])
                self.all_seq_mom.append(traj['seq_mom'][:trajectory_length])

        self.all_seq_qpos = torch.from_numpy(np.array(self.all_seq_qpos))
        self.all_seq_mom = torch.from_numpy(np.array(self.all_seq_mom))
        self.all_seq_torque = torch.from_numpy(np.array(self.all_seq_torque))

        print("---------------Statistics--------------")
        print(f"Range of qpos: [{self.all_seq_qpos.max()} - {self.all_seq_qpos.min()}]; Std of qpos: {self.all_seq_qpos.std()}")
        print(f"Range of mom: [{self.all_seq_mom.max()} - {self.all_seq_mom.min()}]; Std of mom: {self.all_seq_mom.std()}")
        print(f"Range of torque: [{self.all_seq_torque.max()} - {self.all_seq_torque.min()}]; Std of torque: {self.all_seq_torque.std()}")

    def __len__(self):
        return self.num_traj

    def __getitem__(self, index):
        return {
            'seq_qpos': self.all_seq_qpos[index],
            'seq_mom': self.all_seq_mom[index],
            'seq_torque': self.all_seq_torque[index],
        }

if __name__ == '__main__':
    dataset = TrajectoryDPFCached('/home/gsang/Projects/Perceiver_IO/output/generated_trajectories.h5')
    print(dataset[0]['seq_qpos'].shape)