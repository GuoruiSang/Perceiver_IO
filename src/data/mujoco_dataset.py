import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class MujocoDataset(Dataset):
    """
    PyTorch Dataset for loading trajectories from a MuJoCo HDF5 file.

    Each item in the dataset corresponds to a single trajectory, which consists of
    qpos, qvel, and torque sequences.
    """
    def __init__(self, h5_path):
        """
        Args:
            h5_path (str): Path to the HDF5 file.
        """
        self.h5_path = h5_path
        self.h5_file = None  # File handle is managed by __getitem__ for multiprocessing
        self.metadata = {}
        
        with h5py.File(self.h5_path, 'r') as f:
            self.num_trajectories = f['episode/qpos'].shape[0]
            # Load metadata
            for key, value in f['episode'].attrs.items():
                self.metadata[key] = value

    def __len__(self):
        """
        Returns the number of trajectories in the dataset.
        """
        return self.num_trajectories

    def __getitem__(self, idx):
        """
        Retrieves a trajectory's data by index.

        Args:
            idx (int): Index of the trajectory to retrieve.

        Returns:
            dict: A dictionary containing the trajectory data as PyTorch tensors.
                  {'qpos': tensor, 'qvel': tensor, 'torque': tensor}
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        
        qpos = self.h5_file['episode/qpos'][idx]  # [T, 8]
        qvel = self.h5_file['episode/qvel'][idx]  # [T, 6]
        torque = self.h5_file['episode/torque'][idx]  # [T, 6]
        
        trajectory_data = {
            'qpos': torch.from_numpy(qpos).float(),
            'qvel': torch.from_numpy(qvel).float(),
            'torque': torch.from_numpy(torque).float()
        }
        
        return trajectory_data

if __name__ == '__main__':
    # Example usage:
    # Make sure 'mujoco_dataset.h5' is in the same directory or provide the correct path.
    try:
        dataset = MujocoDataset('mujoco_dataset.h5')
        print(f"Number of trajectories: {len(dataset)}")

        # Print metadata
        print("\nMetadata:")
        for key, value in dataset.metadata.items():
            # Truncate long values for readability
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + '...'
            print(f"  {key}: {value}")

        # Get the first trajectory
        first_trajectory = dataset[0]
        print("\nData for the first trajectory:")
        for key, value in first_trajectory.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

        # You can also use it with a DataLoader
        from torch.utils.data import DataLoader
        data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Get one batch
        batch = next(iter(data_loader))
        print("\nBatch data from DataLoader:")
        for key, value in batch.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    except FileNotFoundError:
        print("\n'mujoco_dataset.h5' not found.")
        print("Please generate the dataset first by running 'create_dataset_final.py'.")
