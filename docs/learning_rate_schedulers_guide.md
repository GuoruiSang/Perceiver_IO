# Learning Rate Scheduler Guide

This guide explains the different learning rate schedulers available in the training system and when to use each one.

## Overview

Learning rate scheduling is crucial for training deep learning models effectively. A good schedule can:
- Speed up convergence
- Improve final model performance
- Prevent overfitting
- Enable training with higher initial learning rates

## Available Schedulers

### 1. Warmup + Cosine Annealing ⭐ **Recommended**

**Best for:** Most scenarios, stable training, long training runs

**How it works:**
- **Warmup phase:** Gradually increases LR from near-zero to the base LR over the first few epochs
- **Cosine phase:** Smoothly decreases LR following a cosine curve to a minimum value

**Configuration:**
```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.WARMUP_COSINE,
    max_epochs=1000,
    warmup_epochs=20,      # Warmup for first 20 epochs
    eta_min=1e-7           # Minimum LR at the end
)
```

**Advantages:**
- Stable training from the start (warmup prevents early instability)
- Smooth learning rate decay
- Well-suited for transfer learning and fine-tuning
- Used successfully in many SOTA models (BERT, ViT, etc.)

**When to use:**
- Default choice for most tasks
- When training from scratch
- Long training runs (100+ epochs)
- When stability is important

---

### 2. OneCycleLR

**Best for:** Fast experimentation, shorter training runs, aggressive training

**How it works:**
- Increases LR from initial to max over `pct_start` fraction of training
- Decreases LR from max to final over remaining fraction
- Single cycle over entire training

**Configuration:**
```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.ONE_CYCLE,
    max_epochs=1000,
    steps_per_epoch=len(train_dataloader),
    max_lr=3e-3,           # Peak learning rate
    pct_start=0.3,         # Spend 30% of training increasing LR
    div_factor=25.0,       # Initial LR = max_lr / 25
    final_div_factor=1e4   # Final LR = max_lr / 10000
)
```

**Advantages:**
- Often achieves faster convergence
- Can use higher learning rates safely
- Good regularization effect
- Popular in computer vision tasks

**Disadvantages:**
- Less stable than cosine annealing
- Requires knowing total training steps in advance
- Can overfit if not tuned properly

**When to use:**
- When you want fast convergence
- Shorter training runs (< 100 epochs)
- When you have well-tuned hyperparameters
- Computer vision tasks

---

### 3. Cosine Annealing with Warm Restarts

**Best for:** Avoiding local minima, ensemble-like behavior

**How it works:**
- Periodic cosine annealing with "restarts"
- LR drops then jumps back up periodically
- Each restart cycle can be longer than the previous (controlled by `T_mult`)

**Configuration:**
```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.COSINE_WARM_RESTARTS,
    max_epochs=1000,
    T_0=50,        # First restart at epoch 50
    T_mult=2,      # Each cycle is 2x longer: 50, 100, 200...
    eta_min=1e-7   # Minimum LR in each cycle
)
```

**Advantages:**
- Can escape local minima
- Each restart can find better solutions
- Ensemble-like behavior (can checkpoint at each cycle)
- Good for very long training

**Disadvantages:**
- Can be unstable at restart points
- Requires tuning restart intervals
- May not converge as smoothly

**When to use:**
- Very long training runs (500+ epochs)
- When you suspect local minima issues
- When you want to create ensembles
- Research and experimentation

---

### 4. Reduce on Plateau

**Best for:** Adaptive training, unknown optimal schedule

**How it works:**
- Monitors validation loss
- Reduces LR by a factor when loss stops improving
- Completely adaptive to your training dynamics

**Configuration:**
```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.REDUCE_ON_PLATEAU,
    max_epochs=1000,
    mode='min',      # Minimize the metric
    factor=0.5,      # Reduce LR by half
    patience=15,     # Wait 15 epochs before reducing
    min_lr=1e-7      # Don't go below this
)
```

**Advantages:**
- Completely adaptive
- No need to know training dynamics in advance
- Safe and conservative
- Good for unknown datasets

**Disadvantages:**
- Can be too conservative
- Slower convergence
- Requires validation set
- Less reproducible

**When to use:**
- New datasets with unknown characteristics
- When you don't want to tune schedules
- When validation loss is a reliable metric
- Conservative training approach

---

### 5. Standard Cosine Annealing

**Best for:** Simple, predictable decay

**How it works:**
- Smoothly decreases LR from initial to minimum following a cosine curve
- No warmup, no restarts

**Configuration:**
```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.COSINE_ANNEALING,
    max_epochs=1000,
    eta_min=1e-7     # Minimum LR at the end
)
```

**Advantages:**
- Simple and predictable
- Smooth convergence
- Works well in many cases
- No hyperparameters to tune

**Disadvantages:**
- No warmup (can be unstable early)
- Less flexible than Warmup + Cosine

**When to use:**
- Fine-tuning pretrained models
- When starting from good initialization
- Medium-length training (50-200 epochs)

---

## Comparison Table

| Scheduler | Warmup | Complexity | Stability | Speed | Best For |
|-----------|--------|------------|-----------|-------|----------|
| **Warmup + Cosine** | ✅ Yes | Low | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General use |
| **OneCycleLR** | ✅ Yes | Medium | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Fast training |
| **Cosine Warm Restarts** | ❌ No | High | ⭐⭐⭐ | ⭐⭐⭐ | Long runs |
| **Reduce on Plateau** | ❌ No | Low | ⭐⭐⭐⭐⭐ | ⭐⭐ | Conservative |
| **Cosine Annealing** | ❌ No | Low | ⭐⭐⭐⭐ | ⭐⭐⭐ | Fine-tuning |

---

## Recommendations by Scenario

### Your Current Task (Torque Prediction)

**Recommended:** Warmup + Cosine Annealing

Your task is a regression problem with 1000 epochs of training. The Warmup + Cosine schedule provides:
- Stable initial training (warmup helps with regression tasks)
- Smooth convergence over long training
- Good balance of speed and stability

### If Training is Too Slow

**Try:** OneCycleLR

If you want faster results:
```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.ONE_CYCLE,
    max_epochs=500,  # Reduce epochs
    steps_per_epoch=len(train_dataloader),
    max_lr=5e-3,     # Higher max LR
    pct_start=0.3
)
```

### If Loss is Unstable

**Try:** Reduce on Plateau

For more conservative, adaptive training:
```python
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.REDUCE_ON_PLATEAU,
    max_epochs=1000,
    patience=20,     # Increase patience for stability
    factor=0.7       # Gentler reduction
)
```

---

## Tips and Best Practices

### 1. Learning Rate Range
- **Too low:** Slow convergence, might not reach optimum
- **Too high:** Unstable training, divergence
- **Good starting point:** 1e-4 to 1e-3 for Adam/AdamW

### 2. Warmup Duration
- **Short training (< 50 epochs):** 5-10 epochs
- **Medium training (50-200 epochs):** 10-20 epochs
- **Long training (200+ epochs):** 20-50 epochs
- **Rule of thumb:** 5-10% of total epochs

### 3. Monitoring
- Always monitor learning rate in your logger (WandB, TensorBoard)
- Check if LR schedule matches your expectations
- Look for correlation between LR changes and loss changes

### 4. Optimizer Choice
- **AdamW** (used in implementation) is better than Adam for most cases
- Includes weight decay for better regularization
- Works well with all schedulers

### 5. Combining with Other Techniques
- Early stopping: Use with any scheduler
- Gradient clipping: Especially useful with OneCycleLR
- Mixed precision: Compatible with all schedulers

---

## Visualization of Schedules

Here's what each schedule looks like over 100 epochs (starting LR = 1e-3):

```
Warmup + Cosine (warmup=10):
    │
1e-3│    ╱─────╲
    │   ╱       ╲
    │  ╱         ╲___
1e-7│ ╱              ╲___
    └────────────────────
    0   10   50    100

OneCycleLR (pct_start=0.3):
    │
3e-3│      ╱╲
    │     ╱  ╲
1e-3│    ╱    ╲___
    │___╱         ╲____
1e-7│                  ╲
    └────────────────────
    0   30   50    100

Cosine Warm Restarts (T_0=25):
    │
1e-3│ ╲   ╱╲   ╱╲   ╱╲
    │  ╲ ╱  ╲ ╱  ╲ ╱  ╲
    │   ╳    ╳    ╳    ╳
1e-7│  ╱ ╲  ╱ ╲  ╱ ╲  ╱
    └────────────────────
    0   25  50  75  100
```

---

## Implementation Example

Here's how to switch between schedulers in your code:

```python
# In torque_predictor.py, around line 137-144

# OPTION 1: Warmup + Cosine (Current - Recommended)
scheduler_config = LRSchedulerConfig(
    scheduler_type=SchedulerType.WARMUP_COSINE,
    max_epochs=1000,
    warmup_epochs=20,
    eta_min=1e-7
)

# OPTION 2: OneCycleLR (Uncomment to use)
# scheduler_config = LRSchedulerConfig(
#     scheduler_type=SchedulerType.ONE_CYCLE,
#     max_epochs=1000,
#     steps_per_epoch=len(train_dataloader),
#     max_lr=3e-3,
#     pct_start=0.3,
#     div_factor=25.0,
#     final_div_factor=1e4
# )

model = TorquePredictor(
    qpos_dim,
    torque_dim,
    max_timesteps,
    lr=1e-3,  # Base learning rate
    scheduler_config=scheduler_config
)
```

---

## Troubleshooting

### Training Loss Not Decreasing
- Increase learning rate (try 3e-3 or 5e-3)
- Use OneCycleLR for more aggressive training
- Check if warmup is too short

### Validation Loss Plateauing
- Try ReduceOnPlateau for adaptive reduction
- Increase patience if using ReduceOnPlateau
- Consider that you might have converged (check LR value)

### Loss Spikes/Instability
- Add or extend warmup period
- Reduce peak learning rate
- Use more conservative schedule (Cosine or ReduceOnPlateau)
- Add gradient clipping

### Training Too Slow
- Use OneCycleLR
- Increase base learning rate
- Reduce warmup period
- Check if LR has decayed too quickly

---

## References

- [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120) - OneCycleLR
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) - Cosine Warm Restarts
- [Bag of Tricks for Image Classification with CNNs](https://arxiv.org/abs/1812.01187) - Warmup strategies
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) - AdamW optimizer

