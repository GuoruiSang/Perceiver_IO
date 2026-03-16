import os
import tempfile
import traceback

import pytorch_lightning as pl

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from src.models.utils import (
    visualize_trajectory,
    compare_generated_with_reconstructed,
    compare_multiple_generated_with_reconstructed,
)

import faulthandler
import torch


class WandBTrajectoryCallback(pl.Callback):
    """Log sampled trajectory visualizations to W&B during training."""

    def __init__(
        self,
        log_every_n_epochs: int = 10,
        num_samples: int = 1,
        sampling_torque_policy: str = "mixed_reacher",
        sampling_torque_mix: str = "sinusoidal:0.32,lpf_uniform:0.25,drift_ou:0.43",
        sampling_lpf_uniform_beta: float = 0.9992,
        sampling_torque_scale: float = 0.35,
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.sampling_torque_policy = sampling_torque_policy
        self.sampling_torque_mix = sampling_torque_mix
        self.sampling_lpf_uniform_beta = sampling_lpf_uniform_beta
        self.sampling_torque_scale = sampling_torque_scale

    def _sample_visualization_trajectory(self, trainer, pl_module):
        val_loaders = trainer.val_dataloaders
        if isinstance(val_loaders, (list, tuple)):
            if len(val_loaders) == 0:
                raise RuntimeError("No validation dataloader available for W&B trajectory callback.")
            val_loader = val_loaders[0]
        else:
            val_loader = val_loaders

        batch = next(iter(val_loader))
        device = pl_module.device
        qpos = batch["seq_qpos"][:1].to(device=device, dtype=torch.float32).repeat(self.num_samples, 1, 1)
        mom = batch["seq_mom"][:1].to(device=device, dtype=torch.float32).repeat(self.num_samples, 1, 1)
        torque = batch["seq_torque"][:1].to(device=device, dtype=torch.float32).repeat(self.num_samples, 1, 1)

        horizon = min(int(qpos.shape[1]), int(pl_module.max_timesteps))
        if horizon < 2:
            raise RuntimeError(f"Need at least 2 timesteps for W&B trajectory visualization, got horizon={horizon}.")

        metadata = {
            "mode": "generic_full_trajectory",
            "prefix_len": None,
            "horizon": horizon,
        }

        sample_kwargs = dict(
            num_samples=self.num_samples,
            trajectory_length=horizon,
            num_diffusion_steps=100,
            context_fraction=0.5,
            use_ema=True,
            sampler="ddim",
            sampling_torque_policy=self.sampling_torque_policy,
            sampling_torque_mix=self.sampling_torque_mix,
            sampling_lpf_uniform_beta=self.sampling_lpf_uniform_beta,
            sampling_torque_scale=self.sampling_torque_scale,
        )
        if getattr(pl_module, "query_context_mode", "random_subset") == "clean_prefix_noisy_suffix":
            prefix_len = max(1, min(horizon - 1, int(round(0.5 * horizon))))
            metadata["mode"] = "observed_prefix_completion"
            metadata["prefix_len"] = prefix_len
            sample_kwargs.update(
                sample_mode="observed_prefix_completion",
                prefix_len=prefix_len,
                observed_qpos=qpos,
                observed_mom=mom,
                observed_torque=torque,
            )

        state, rollout_torque = pl_module.sample_trajectories(**sample_kwargs)
        return state, rollout_torque, metadata

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1

        if not trainer.is_global_zero:
            return
        if not WANDB_AVAILABLE:
            print("[W&B Callback] wandb not available, skipping")
            return
        if trainer.logger is None:
            print("[W&B Callback] trainer.logger is None, skipping")
            return
        if current_epoch % self.log_every_n_epochs != 0:
            return

        try:
            state, torque, sample_meta = self._sample_visualization_trajectory(trainer, pl_module)

            qpos_dim = pl_module.qpos_dim
            mom_dim = pl_module.mom_dim

            state_np = state.detach().cpu()
            torque_np = torque.detach().cpu()
            generated_list = []
            for sample_idx in range(state_np.shape[0]):
                state_traj = state_np[sample_idx]
                torque_traj = torque_np[sample_idx]
                qpos_traj = state_traj[:, :qpos_dim].cpu()
                generated_list.append(
                    {
                        "seq_qpos": qpos_traj,
                        "seq_mom": state_traj[:, qpos_dim:qpos_dim + mom_dim],
                        "seq_torque": torque_traj,
                    }
                )

            with tempfile.TemporaryDirectory() as tmp_dir:
                if pl_module.xml_content is not None:
                    xml_path = os.path.join(tmp_dir, "model.xml")
                    with open(xml_path, "w") as f:
                        f.write(pl_module.xml_content)

                    faulthandler.dump_traceback_later(60, repeat=False)
                    prefix_len = int(sample_meta.get("prefix_len") or 0)
                    if len(generated_list) > 1:
                        compare_multiple_generated_with_reconstructed(
                            generated_list=generated_list,
                            mujoco_model_path=xml_path,
                            save_path=tmp_dir,
                            dt=pl_module.dt,
                            data_dt=pl_module.data_dt,
                            name="comparison",
                            prefix_len=prefix_len,
                            qpos_representation=pl_module.qpos_representation,
                        )
                    else:
                        compare_generated_with_reconstructed(
                            generated_list[0],
                            xml_path,
                            tmp_dir,
                            dt=pl_module.dt,
                            data_dt=pl_module.data_dt,
                            name="comparison",
                            prefix_len=prefix_len,
                            qpos_representation=pl_module.qpos_representation,
                        )
                    faulthandler.cancel_dump_traceback_later()
                    plot_path = os.path.join(tmp_dir, "comparison.jpg")
                else:
                    fallback_traj = generated_list[0].copy()
                    fallback_traj["seq_qpos"] = pl_module.decode_qpos(fallback_traj["seq_qpos"]).cpu()
                    visualize_trajectory(fallback_traj, tmp_dir)
                    plot_path = os.path.join(tmp_dir, "trajectory.jpg")

                wandb.log({
                    "sampled_trajectory": wandb.Image(plot_path),
                })

        except Exception as e:
            print(f"[W&B] Failed to log sample trajectory: {e}")
            traceback.print_exc()
