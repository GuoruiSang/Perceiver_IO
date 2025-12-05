# HNN Documentation Suite

This directory contains comprehensive documentation for the Hamiltonian Neural Network (HNN) implementation in `src/models/HNN.py`.

---

## 📚 Documentation Files

### 1. **HNN_Documentation.md** - Complete Reference
**Best for**: Deep understanding of theory and implementation

**Contents**:
- 📖 Introduction to Hamiltonian Neural Networks
- 🔬 Mathematical foundations (Hamilton's equations, SO(3) rotations)
- 💻 Detailed code walkthrough (line-by-line explanations)
- 🎯 Training process and loss functions
- 🛠️ Usage guide with examples
- 🐛 Troubleshooting section
- 🎓 Advanced topics and extensions
- 📚 References and further reading
- 📝 Comprehensive glossary

**Read this if**: You want to fully understand why and how everything works.

---

### 2. **HNN_Quick_Reference.md** - TL;DR Guide
**Best for**: Quick lookups and getting started fast

**Contents**:
- 🎯 What does this code do? (plain English)
- 🏗️ Architecture overview (simplified)
- 📊 Data flow diagrams
- 🔑 Key components summary
- 🚀 Usage examples (copy-paste ready)
- 🐛 Common issues & solutions
- 📈 Performance expectations

**Read this if**: You just want to run experiments and understand the basics.

---

### 3. **HNN_Visual_Guide.md** - Diagrams & Flowcharts
**Best for**: Visual learners and understanding system architecture

**Contents**:
- 📦 Module dependency graphs
- 🔄 Data preprocessing pipeline (step-by-step)
- 🧠 Neural network architecture diagram
- 🔁 Training loop flowchart
- 🎯 Inference process visualization
- 📊 Loss function breakdown
- 🔄 Rotation representation conversions
- 🎪 Complete system diagram

**Read this if**: You learn best from diagrams and visual representations.

---

## 🎯 Which Document Should I Read?

### I'm a complete beginner to HNNs
**Start here**: 
1. `HNN_Quick_Reference.md` - Section "What Does This Code Do?"
2. `HNN_Documentation.md` - Sections "Introduction" and "What is a Hamiltonian Neural Network?"
3. Run: `python HNN.py --max-trajectories 100 --epochs 5`
4. `HNN_Visual_Guide.md` - Look at the diagrams

**Time needed**: 30-60 minutes

---

### I understand NNs but not physics
**Start here**:
1. `HNN_Documentation.md` - Section "Mathematical Foundation"
2. `HNN_Visual_Guide.md` - "SO(3) Rotation Representation"
3. `HNN_Quick_Reference.md` - "Mathematics Cheat Sheet"

**Time needed**: 1-2 hours

---

### I understand physics but not the code
**Start here**:
1. `HNN_Quick_Reference.md` - Section "Key Components"
2. `HNN_Visual_Guide.md` - "System Overview" and "Data Flow"
3. `HNN_Documentation.md` - Section "Detailed Component Explanation"

**Time needed**: 1-2 hours

---

### I just want to run it NOW
**Start here**:
1. `HNN_Quick_Reference.md` - Section "Usage Examples"
2. Copy-paste a command:
   ```bash
   python HNN.py --max-trajectories 1000 --epochs 10
   ```
3. Wait for results (~15 minutes)
4. Check output: `hnn_runs/torque_true_vs_pred.png`

**Time needed**: 2 minutes to start, 15 minutes to complete

---

### I want to modify the code
**Start here**:
1. `HNN_Documentation.md` - Section "Code Architecture" + "Detailed Component Explanation"
2. `HNN_Visual_Guide.md` - "Module Dependency Graph"
3. `HNN_Quick_Reference.md` - "Key Components" table

**Time needed**: 2-4 hours

---

### I want to use this for my research
**Read all of them!** Specifically:
1. `HNN_Documentation.md` - Everything, especially "Mathematical Foundation" and "Extensions & Future Work"
2. `HNN_Visual_Guide.md` - "Comparison: HNN vs Alternatives"
3. `HNN_Documentation.md` - Section "References & Further Reading"

**Time needed**: 4-8 hours

---

## 📖 Suggested Reading Order

### For Students/Beginners

```
1. HNN_Quick_Reference.md (30 min)
   └─ Get the big picture
      │
2. HNN_Visual_Guide.md (30 min)
   └─ See how it works visually
      │
3. Run a quick experiment (15 min)
   └─ python HNN.py --max-trajectories 100 --epochs 5
      │
4. HNN_Documentation.md (2-3 hours)
   └─ Deep dive into theory and implementation
      │
5. Experiment with modifications (ongoing)
```

---

### For Researchers/Practitioners

```
1. HNN_Quick_Reference.md - "What Does This Code Do?" (5 min)
   └─ Understand the goal
      │
2. HNN_Documentation.md - "Mathematical Foundation" (30 min)
   └─ Verify the approach is sound
      │
3. HNN_Visual_Guide.md - "Complete System Diagram" (10 min)
   └─ See the full pipeline
      │
4. HNN_Documentation.md - "Detailed Component Explanation" (1 hour)
   └─ Understand implementation details
      │
5. Run full training (2-3 hours)
   └─ python HNN.py --epochs 30
      │
6. HNN_Documentation.md - "Extensions & Future Work" (30 min)
   └─ Plan your research direction
```

---

## 🔑 Key Concepts Summary

If you only remember a few things, remember these:

### 1. **What is an HNN?**
A neural network that learns an energy function (Hamiltonian) instead of direct predictions. This forces the model to respect physics laws like energy conservation.

### 2. **Why use an HNN?**
- ✅ More data-efficient than black-box NNs
- ✅ Physically consistent predictions
- ✅ Can solve inverse problems (crucial for control)
- ✅ Better extrapolation to unseen scenarios

### 3. **How does it work?**
```
1. Network learns: H(q, p) = energy
2. Autograd computes: ∂H/∂q and ∂H/∂p
3. Physics applies: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q + control
```

### 4. **What are the inputs/outputs?**
- **Input**: Robot state (position q, momentum p)
- **Output**: Energy H
- **Derived**: Velocities and forces (via derivatives)
- **Application**: Predict required torques for desired motion

### 5. **What's the training process?**
```
For each trajectory:
  1. Preprocess: Convert quaternions → rotation vectors
  2. Compute: mass matrices, momenta from MuJoCo
  3. Train: Match predicted derivatives to true derivatives
  4. Result: Network learns energy landscape
```

---

## 🎯 Quick Command Reference

```bash
# Quick test (fast)
python HNN.py --max-trajectories 1000 --epochs 10

# Medium run (balanced)
python HNN.py --max-trajectories 5000 --epochs 20

# Full training (best results)
python HNN.py --epochs 30

# Custom settings
python HNN.py \
  --h5 /path/to/data.h5 \
  --xml /path/to/robot.xml \
  --epochs 30 \
  --batch_size 256 \
  --lr 1e-4 \
  --cache-dir ./my_cache
```

---

## 🐛 Getting Help

### If you're confused about...

**Theory (what/why)**:
- Read: `HNN_Documentation.md` - "Introduction" section
- Watch: Search YouTube for "Hamiltonian Neural Networks"

**Implementation (how)**:
- Read: `HNN_Documentation.md` - "Detailed Component Explanation"
- Visual: `HNN_Visual_Guide.md` - corresponding diagrams

**Usage (running it)**:
- Read: `HNN_Quick_Reference.md` - "Usage Examples"
- Try: Start with small dataset (`--max-trajectories 100`)

**Errors (troubleshooting)**:
- Read: `HNN_Quick_Reference.md` - "Common Issues & Solutions"
- Read: `HNN_Visual_Guide.md` - "Debugging Visualization"

---

## 📊 Documentation Statistics

| File | Lines | Topics Covered | Read Time |
|------|-------|----------------|-----------|
| `HNN_Documentation.md` | ~900 | 14 major sections | 2-3 hours |
| `HNN_Quick_Reference.md` | ~500 | 11 quick topics | 30-45 min |
| `HNN_Visual_Guide.md` | ~750 | 12 visual sections | 30-45 min |
| **Total** | **~2150** | **37 topics** | **3-4 hours** |

---

## 🌟 What Makes This Documentation Special?

1. **Three Different Formats**: Theory, quick reference, and visual - pick what works for you
2. **Beginner-Friendly**: Assumes no prior knowledge of Hamiltonian mechanics
3. **Comprehensive**: Covers theory, implementation, usage, and troubleshooting
4. **Practical**: Real command examples and expected outputs
5. **Visual**: Lots of diagrams and flowcharts
6. **Searchable**: Detailed table of contents in each document

---

## 🎓 Learning Resources Beyond This Documentation

### Papers
- **Original HNN**: Greydanus et al., "Hamiltonian Neural Networks" (NeurIPS 2019)
- **Lagrangian NNs**: Cranmer et al., "Lagrangian Neural Networks" (ICLR 2020)
- **Neural ODEs**: Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)

### Books
- **Classical Mechanics**: Goldstein, "Classical Mechanics" (Chapter on Hamiltonian formulation)
- **Robot Dynamics**: Murray, Li, Sastry, "A Mathematical Introduction to Robotic Manipulation"
- **Deep Learning**: Goodfellow et al., "Deep Learning" (MIT Press)

### Online
- **MuJoCo Docs**: https://mujoco.readthedocs.io/
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/
- **SO(3) Math**: https://ethaneade.com/lie.pdf

---

## 📝 How to Cite

If you use this code or documentation in your research, please consider citing:

```bibtex
@software{hnn_implementation,
  title={Hamiltonian Neural Network for Robotic Dynamics},
  author={[Your Name]},
  year={2025},
  url={[Your Repository URL]}
}
```

And the original HNN paper:

```bibtex
@inproceedings{greydanus2019hamiltonian,
  title={Hamiltonian neural networks},
  author={Greydanus, Samuel and Dzamba, Misko and Yosinski, Jason},
  booktitle={Advances in Neural Information Processing Systems},
  pages={15379--15389},
  year={2019}
}
```

---

## 🔄 Documentation Updates

**Current Version**: 1.0  
**Last Updated**: November 2025

**Change Log**:
- v1.0 (Nov 2025): Initial comprehensive documentation release
  - Created three complementary documentation files
  - Added visual diagrams and flowcharts
  - Included troubleshooting sections
  - Added beginner-friendly explanations

---

## 💬 Feedback

Found an error? Have a suggestion? Want a specific topic explained better?

Please open an issue or submit a pull request!

---

## 🎉 Acknowledgments

This documentation was created to make Hamiltonian Neural Networks accessible to everyone, from beginners to experts. Special thanks to:

- **Greydanus et al.** for the original HNN paper
- **MuJoCo team** for the excellent physics simulator
- **PyTorch Lightning** for the clean training framework
- **You** for taking the time to learn!

---

Happy learning! 🚀🤖

*"The best way to understand physics is to learn the energy function."*  
— Every Hamiltonian enthusiast

