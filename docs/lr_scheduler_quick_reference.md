# Learning Rate Scheduler Quick Reference

Quick copy-paste configurations for different scenarios.

## 🎯 Just Want the Best Default?

Use **Warmup + Cosine Annealing**:

```python
from src.training import LRSchedulerConfig, SchedulerType

scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.WARMUP_COSINE,
    max_epochs=1000,
    warmup_epochs=20,
    eta_min=1e-7
)

model = TorquePredictor(
    qpos_dim, torque_dim, max_timesteps,
    lr=1e-3,
    scheduler_config=scheduler_config
)
```

---

## 🚀 Want Fastest Convergence?

Use **OneCycleLR**:

```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.ONE_CYCLE,
    max_epochs=500,  # Can often train with fewer epochs
    steps_per_epoch=len(train_dataloader),
    max_lr=3e-3,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4
)

model = TorquePredictor(
    qpos_dim, torque_dim, max_timesteps,
    lr=1e-3,  # OneCycle will override this
    scheduler_config=scheduler_config
)
```

---

## 🛡️ Want Most Stable Training?

Use **Reduce on Plateau**:

```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.REDUCE_ON_PLATEAU,
    max_epochs=1000,
    mode='min',
    factor=0.5,
    patience=20,
    min_lr=1e-7
)

model = TorquePredictor(
    qpos_dim, torque_dim, max_timesteps,
    lr=1e-3,
    scheduler_config=scheduler_config
)
```

---

## 🔄 Want to Escape Local Minima?

Use **Cosine Warm Restarts**:

```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.COSINE_WARM_RESTARTS,
    max_epochs=1000,
    T_0=50,
    T_mult=2,
    eta_min=1e-7
)

model = TorquePredictor(
    qpos_dim, torque_dim, max_timesteps,
    lr=1e-3,
    scheduler_config=scheduler_config
)
```

---

## 📉 Want Simple Decay?

Use **Cosine Annealing**:

```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.COSINE_ANNEALING,
    max_epochs=1000,
    eta_min=1e-7
)

model = TorquePredictor(
    qpos_dim, torque_dim, max_timesteps,
    lr=1e-3,
    scheduler_config=scheduler_config
)
```

---

## ⚡ Quick Decision Tree

```
START
  │
  ├─ Training from scratch? ──YES──> Warmup + Cosine ⭐
  │                          
  ├─ Need fast results? ──YES──> OneCycleLR 🚀
  │
  ├─ Unstable training? ──YES──> Reduce on Plateau 🛡️
  │
  ├─ Very long training (500+ epochs)? ──YES──> Cosine Warm Restarts 🔄
  │
  └─ Fine-tuning pretrained? ──YES──> Cosine Annealing 📉
```

---

## 📊 Learning Rate Recommendations

| Base LR | Use Case |
|---------|----------|
| `1e-4` | Conservative, stable training |
| `1e-3` | Standard choice (recommended) ⭐ |
| `3e-3` | Aggressive, faster convergence |
| `1e-2` | Very aggressive (OneCycle only) |

---

## 🎛️ Warmup Recommendations

| Total Epochs | Warmup Epochs |
|--------------|---------------|
| 50-100 | 5-10 |
| 100-500 | 10-20 |
| 500-1000 | 20-50 |
| 1000+ | 50-100 |

**Rule of thumb:** 5-10% of total epochs

---

## 🔧 Troubleshooting

### Training Loss Not Decreasing
```python
# Increase learning rate
lr=3e-3  # or even 5e-3

# Or use more aggressive schedule
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.ONE_CYCLE,
    max_epochs=500,
    steps_per_epoch=len(train_dataloader),
    max_lr=5e-3  # Higher!
)
```

### Loss Exploding
```python
# Decrease learning rate
lr=1e-4  # or even 1e-5

# Add/extend warmup
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.WARMUP_COSINE,
    max_epochs=1000,
    warmup_epochs=50  # Longer warmup
)
```

### Validation Loss Plateauing
```python
# Use adaptive schedule
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.REDUCE_ON_PLATEAU,
    max_epochs=1000,
    patience=10,
    factor=0.5
)
```

### Loss Spikes During Training
```python
# More conservative schedule
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.COSINE_ANNEALING,
    max_epochs=1000,
    eta_min=1e-7
)
lr=5e-4  # Lower base LR
```

---

## 📝 Common Patterns

### Pattern 1: Quick Experimentation
```python
# Short training, fast results
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.ONE_CYCLE,
    max_epochs=50,
    steps_per_epoch=len(train_dataloader),
    max_lr=5e-3
)
```

### Pattern 2: Production Training
```python
# Long, stable training
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.WARMUP_COSINE,
    max_epochs=1000,
    warmup_epochs=20,
    eta_min=1e-7
)
```

### Pattern 3: Research/Unknown Dataset
```python
# Adaptive, conservative
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.REDUCE_ON_PLATEAU,
    max_epochs=1000,
    patience=20,
    factor=0.7
)
```

---

## 🔍 Monitoring Your Schedule

Always add the learning rate monitor:

```python
from src.training import LearningRateMonitor

callbacks = [
    ModelCheckpoint(...),
    LearningRateMonitor(logging_interval="epoch")
]

trainer = pl.Trainer(callbacks=callbacks)
```

Then check in WandB/TensorBoard that:
- ✅ LR starts low (if using warmup)
- ✅ LR increases during warmup
- ✅ LR decreases smoothly
- ✅ LR reaches minimum at end

---

## 📈 Visualize Before Training

Run visualization to see your schedule:

```bash
cd /home/gsang/Projects/Perceiver_IO
python src/training/visualize_lr_schedules.py
```

Or programmatically:

```python
from src.training.visualize_lr_schedules import SchedulerVisualizer

visualizer = SchedulerVisualizer(max_epochs=1000)
visualizer.plot_custom_schedule(
    your_config,
    base_lr=1e-3,
    save_path="my_schedule.png"
)
```

---

## 💡 Pro Tips

1. **Start Conservative**: Begin with Warmup + Cosine, tune later
2. **Monitor LR**: Always use LearningRateMonitor callback
3. **Visualize First**: Run visualization before long training
4. **Log Everything**: Track LR, train loss, val loss together
5. **Don't Mix**: Choose one scheduler and tune it, don't randomly switch
6. **Trust the Process**: Modern schedules work well, avoid micro-managing

---

## 🎓 Learn More

- Full guide: `docs/learning_rate_schedulers_guide.md`
- Implementation: `src/training/README.md`
- Examples: See `torque_predictor.py` lines 137-191

