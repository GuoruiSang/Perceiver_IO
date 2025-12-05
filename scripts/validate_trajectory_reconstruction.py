"""
Validation script for verifying trajectory reconstruction from initial conditions and torques.

This script validates that trajectories can be perfectly reconstructed from:
- Initial position (qpos[0])
- Initial velocity (qvel[0])
- Sequence of torques

Using the same MuJoCo configuration that generated the trajectory.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import h5py
import numpy as np
import mujoco
import tempfile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ReconstructionResult:
    """Results from trajectory reconstruction validation."""
    trajectory_index: int
    is_consistent: bool
    max_qpos_error: float
    max_qvel_error: float
    mean_qpos_error: float
    mean_qvel_error: float
    std_qpos_error: float
    std_qvel_error: float
    qpos_errors: np.ndarray
    qvel_errors: np.ndarray


class TrajectoryLoader:
    """Handles loading trajectory data from HDF5 files."""
    
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.h5_file = None
        self.episode = None
        
    def __enter__(self):
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.episode = self.h5_file['episode']
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5_file:
            self.h5_file.close()
    
    def get_num_trajectories(self) -> int:
        """Get total number of trajectories in dataset."""
        return self.episode['qpos'].shape[0]
    
    def get_trajectory(self, index: int) -> Dict[str, np.ndarray]:
        """Load a single trajectory by index."""
        return {
            'qpos': self.episode['qpos'][index],
            'qvel': self.episode['qvel'][index],
            'torque': self.episode['torque'][index]
        }
    
    def get_metadata(self) -> Dict[str, any]:
        """Extract metadata and configuration from dataset."""
        metadata = {}
        for key in self.episode.attrs.keys():
            metadata[key] = self.episode.attrs[key]
        return metadata


class MuJoCoSimulator:
    """Handles MuJoCo simulation setup and trajectory reconstruction."""
    
    def __init__(self, model_xml: str, timestep: float, integrator: str = 'rk4'):
        self.model_xml = model_xml
        self.timestep = timestep
        self.integrator = integrator
        self.model = None
        self.data = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize MuJoCo model with specified configuration."""
        # Create temporary XML file if model_xml is string content
        if self.model_xml.strip().startswith('<'):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(self.model_xml)
                temp_path = f.name
            self.model = mujoco.MjModel.from_xml_path(temp_path)
            Path(temp_path).unlink()  # Clean up temp file
        else:
            self.model = mujoco.MjModel.from_xml_path(self.model_xml)
        
        # Set timestep
        self.model.opt.timestep = self.timestep
        
        # Set integrator
        if self.integrator.lower() == 'rk4':
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        else:
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        
        # Initialize data
        self.data = mujoco.MjData(self.model)
    
    def reconstruct_trajectory(
        self,
        initial_qpos: np.ndarray,
        initial_qvel: np.ndarray,
        torque_sequence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct trajectory from initial conditions and torque sequence.
        
        Args:
            initial_qpos: Initial position
            initial_qvel: Initial velocity
            torque_sequence: Sequence of control torques [timesteps x nv]
        
        Returns:
            Tuple of (reconstructed_qpos, reconstructed_qvel)
        """
        # Reset to initial state
        self.data.qpos[:] = initial_qpos
        self.data.qvel[:] = initial_qvel
        mujoco.mj_forward(self.model, self.data)
        
        num_steps = len(torque_sequence)
        reconstructed_qpos = np.zeros((num_steps, len(initial_qpos)))
        reconstructed_qvel = np.zeros((num_steps, len(initial_qvel)))
        
        for step in range(num_steps):
            # Apply control
            self.data.ctrl[:] = torque_sequence[step]
            
            # Record state before stepping
            reconstructed_qpos[step] = self.data.qpos.copy()
            reconstructed_qvel[step] = self.data.qvel.copy()
            
            # Step simulation forward
            mujoco.mj_step(self.model, self.data)
        
        return reconstructed_qpos, reconstructed_qvel


class TrajectoryValidator:
    """Validates trajectory reconstruction accuracy."""
    
    def __init__(self, qpos_tolerance: float = 1e-6, qvel_tolerance: float = 1e-6):
        self.qpos_tolerance = qpos_tolerance
        self.qvel_tolerance = qvel_tolerance
    
    def validate(
        self,
        original_qpos: np.ndarray,
        original_qvel: np.ndarray,
        reconstructed_qpos: np.ndarray,
        reconstructed_qvel: np.ndarray,
        trajectory_index: int
    ) -> ReconstructionResult:
        """
        Compare original and reconstructed trajectories.
        
        Args:
            original_qpos: Original position sequence
            original_qvel: Original velocity sequence
            reconstructed_qpos: Reconstructed position sequence
            reconstructed_qvel: Reconstructed velocity sequence
            trajectory_index: Index of trajectory being validated
        
        Returns:
            ReconstructionResult with detailed error statistics
        """
        # Compute errors
        qpos_errors = np.abs(original_qpos - reconstructed_qpos)
        qvel_errors = np.abs(original_qvel - reconstructed_qvel)
        
        # Statistics
        max_qpos_error = np.max(qpos_errors)
        max_qvel_error = np.max(qvel_errors)
        mean_qpos_error = np.mean(qpos_errors)
        mean_qvel_error = np.mean(qvel_errors)
        std_qpos_error = np.std(qpos_errors)
        std_qvel_error = np.std(qvel_errors)
        
        # Check if reconstruction is consistent
        is_consistent = (
            max_qpos_error < self.qpos_tolerance and
            max_qvel_error < self.qvel_tolerance
        )
        
        return ReconstructionResult(
            trajectory_index=trajectory_index,
            is_consistent=is_consistent,
            max_qpos_error=max_qpos_error,
            max_qvel_error=max_qvel_error,
            mean_qpos_error=mean_qpos_error,
            mean_qvel_error=mean_qvel_error,
            std_qpos_error=std_qpos_error,
            std_qvel_error=std_qvel_error,
            qpos_errors=qpos_errors,
            qvel_errors=qvel_errors
        )


class ValidationReporter:
    """Handles reporting and displaying validation results."""
    
    @staticmethod
    def print_single_result(result: ReconstructionResult):
        """Print results for a single trajectory."""
        print(f"\nTrajectory {result.trajectory_index}:")
        print(f"  Status: {'✅ PASS' if result.is_consistent else '❌ FAIL'}")
        print(f"  Position Error:")
        print(f"    Max:  {result.max_qpos_error:.2e}")
        print(f"    Mean: {result.mean_qpos_error:.2e}")
        print(f"    Std:  {result.std_qpos_error:.2e}")
        print(f"  Velocity Error:")
        print(f"    Max:  {result.max_qvel_error:.2e}")
        print(f"    Mean: {result.mean_qvel_error:.2e}")
        print(f"    Std:  {result.std_qvel_error:.2e}")
    
    @staticmethod
    def print_summary(results: List[ReconstructionResult]):
        """Print summary statistics for multiple trajectories."""
        num_trajectories = len(results)
        num_passed = sum(1 for r in results if r.is_consistent)
        pass_rate = 100.0 * num_passed / num_trajectories
        
        max_qpos_errors = [r.max_qpos_error for r in results]
        max_qvel_errors = [r.max_qvel_error for r in results]
        mean_qpos_errors = [r.mean_qpos_error for r in results]
        mean_qvel_errors = [r.mean_qvel_error for r in results]
        
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"\nTrajectories validated: {num_trajectories}")
        print(f"Passed: {num_passed} ({pass_rate:.1f}%)")
        print(f"Failed: {num_trajectories - num_passed} ({100.0 - pass_rate:.1f}%)")
        
        print(f"\nPosition Error Statistics:")
        print(f"  Max across all:  {np.max(max_qpos_errors):.2e}")
        print(f"  Mean of maxes:   {np.mean(max_qpos_errors):.2e} ± {np.std(max_qpos_errors):.2e}")
        print(f"  Mean of means:   {np.mean(mean_qpos_errors):.2e} ± {np.std(mean_qpos_errors):.2e}")
        
        print(f"\nVelocity Error Statistics:")
        print(f"  Max across all:  {np.max(max_qvel_errors):.2e}")
        print(f"  Mean of maxes:   {np.mean(max_qvel_errors):.2e} ± {np.std(max_qvel_errors):.2e}")
        print(f"  Mean of means:   {np.mean(mean_qvel_errors):.2e} ± {np.std(mean_qvel_errors):.2e}")
        
        if pass_rate == 100.0:
            print(f"\n🎉 ALL TRAJECTORIES PASSED! Perfect reconstruction!")
        elif pass_rate >= 95.0:
            print(f"\n✓ Most trajectories passed ({pass_rate:.1f}%)")
        else:
            print(f"\n⚠ Warning: {100.0 - pass_rate:.1f}% of trajectories failed validation")
        
        print("=" * 80)


class ValidationOrchestrator:
    """Orchestrates the validation process."""
    
    def __init__(
        self,
        trajectory_path: str,
        model_xml_path: Optional[str] = None,
        qpos_tolerance: float = 1e-6,
        qvel_tolerance: float = 1e-6
    ):
        self.trajectory_path = trajectory_path
        self.model_xml_path = model_xml_path
        self.qpos_tolerance = qpos_tolerance
        self.qvel_tolerance = qvel_tolerance
        
        self.validator = TrajectoryValidator(qpos_tolerance, qvel_tolerance)
        self.reporter = ValidationReporter()
    
    def validate_trajectory(self, index: int) -> ReconstructionResult:
        """Validate a single trajectory."""
        with TrajectoryLoader(self.trajectory_path) as loader:
            # Load metadata and trajectory
            metadata = loader.get_metadata()
            trajectory = loader.get_trajectory(index)
            
            # Determine model XML
            if self.model_xml_path:
                model_xml = self.model_xml_path
            elif 'robot_arm_xml' in metadata:
                model_xml = metadata['robot_arm_xml']
            else:
                raise ValueError("No model XML found in dataset or provided as argument")
            
            # Get simulation parameters
            timestep = metadata.get('step_time', 0.001)
            integrator = 'rk4'  # Default, can be extracted from metadata if available
            
            # Initialize simulator
            simulator = MuJoCoSimulator(model_xml, timestep, integrator)
            
            # Extract components
            initial_qpos = trajectory['qpos'][0]
            initial_qvel = trajectory['qvel'][0]
            torque_sequence = trajectory['torque']
            
            # Reconstruct trajectory
            reconstructed_qpos, reconstructed_qvel = simulator.reconstruct_trajectory(
                initial_qpos, initial_qvel, torque_sequence
            )
            
            # Validate
            result = self.validator.validate(
                trajectory['qpos'],
                trajectory['qvel'],
                reconstructed_qpos,
                reconstructed_qvel,
                index
            )
            
            return result
    
    def validate_all(self, max_trajectories: Optional[int] = None) -> List[ReconstructionResult]:
        """Validate all trajectories in dataset."""
        with TrajectoryLoader(self.trajectory_path) as loader:
            num_trajectories = loader.get_num_trajectories()
        
        if max_trajectories:
            num_trajectories = min(num_trajectories, max_trajectories)
        
        print(f"Validating {num_trajectories} trajectories...")
        
        results = []
        for i in range(num_trajectories):
            result = self.validate_trajectory(i)
            results.append(result)
            
            # Print progress every 10 trajectories
            if (i + 1) % 10 == 0 or i == num_trajectories - 1:
                print(f"Progress: {i+1}/{num_trajectories}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate trajectory reconstruction from initial conditions and torques"
    )
    parser.add_argument(
        "--trajectory_file",
        type=str,
        default="/home/gsang/Projects/Perceiver_IO/data/mujoco_dataset_final.h5",
        help="Path to HDF5 file containing trajectories"
    )
    parser.add_argument(
        "--model_xml",
        type=str,
        default="/home/gsang/Projects/Perceiver_IO/configs/robotic_arm.xml",
        help="Path to MuJoCo XML model (if not stored in dataset)"
    )
    parser.add_argument(
        "--trajectory_index",
        type=int,
        default=None,
        help="Specific trajectory index to validate (default: validate all)"
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to validate (default: all)"
    )
    parser.add_argument(
        "--qpos_tolerance",
        type=float,
        default=1e-6,
        help="Maximum acceptable position error"
    )
    parser.add_argument(
        "--qvel_tolerance",
        type=float,
        default=1e-6,
        help="Maximum acceptable velocity error"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Print detailed results for each trajectory"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAJECTORY RECONSTRUCTION VALIDATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Trajectory file: {args.trajectory_file}")
    print(f"  Model XML: {args.model_xml or '(from dataset metadata)'}")
    print(f"  Position tolerance: {args.qpos_tolerance:.2e}")
    print(f"  Velocity tolerance: {args.qvel_tolerance:.2e}")
    
    # Initialize orchestrator
    orchestrator = ValidationOrchestrator(
        args.trajectory_file,
        args.model_xml,
        args.qpos_tolerance,
        args.qvel_tolerance
    )
    
    # Validate
    if args.trajectory_index is not None:
        # Validate single trajectory
        print(f"\nValidating trajectory {args.trajectory_index}...")
        result = orchestrator.validate_trajectory(args.trajectory_index)
        orchestrator.reporter.print_single_result(result)
    else:
        # Validate multiple trajectories
        results = orchestrator.validate_all(args.max_trajectories)
        
        # Print individual results if verbose
        if args.verbose:
            for result in results:
                orchestrator.reporter.print_single_result(result)
        
        # Print summary
        orchestrator.reporter.print_summary(results)


if __name__ == "__main__":
    main()

