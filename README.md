# CartPole Balancing with PPO (Stable-Baselines3)

This repository contains a clean and modern implementation of a Reinforcement Learning (RL) agent trained to solve the classic **CartPole-v1** environment using the **Proximal Policy Optimization (PPO)** algorithm.

The project leverages the updated **Gymnasium** library (by Farama Foundation) and **Stable-Baselines3** for reliable RL baselines.

---

## 🛠️ Tech Stack & Features

*   **Environment:** [Gymnasium](https://gymnasium.farama.org/) `CartPole-v1` (State-space: 4 continuous variables; Action-space: 2 discrete actions).
*   **RL Framework:** [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO Implementation).
*   **Deep Learning Backend:** PyTorch.
*   **Features:** 
    *   Automated training pipeline with real-time logging.
    *   Model serialization (saving and loading `.zip` weights).
    *   Post-training evaluation utilizing robust `evaluate_policy` wrappers.
    *   Human-render mode for visual performance tracking.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/cartpole-ppo-rl.git](https://github.com/YOUR_USERNAME/cartpole-ppo-rl.git)
cd cartpole-ppo-rl
