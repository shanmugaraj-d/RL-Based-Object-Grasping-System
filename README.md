# 🤖 RL-Based Object Grasping System

A Machine Learning Lab project to train a robotic arm in a simulated PyBullet environment to grasp objects of various shapes and sizes using Reinforcement Learning.

---

## 📁 Project Overview

This project demonstrates how a robotic arm learns to grasp objects like cubes, spheres, and cylinders placed randomly on a table. Using PyBullet for simulation and a structured reward system, the model learns to perform efficient and successful grasps in diverse environments.

---

## 🧠 Key Features

- ✅ **Reinforcement Learning (RL)**-based training loop
- 🏗️ Simulated environment using **PyBullet**
- 📦 Handles multiple object types: **Cube, Sphere, Cylinder**
- 🧾 Reward shaping to encourage speed and accuracy
- 📊 Logs success rate, time steps, and cumulative reward
- 🔁 Environment randomization for robust generalization

---

## 📚 Methodology

### Simulation

- PyBullet simulates robotic interactions in a safe, resettable environment.
- Object type, size, and position randomized per episode.

### Reward System

- **+10** reward for reaching within `0.05m` of the object  
- **Inverse distance reward** as the robot moves closer  
- **-0.01** per time step to encourage faster actions  
- **-5** penalty for invalid or far-off movements  

### State Vector

- Distance from end-effector to object  
- One-hot encoding for shape  
- Object size  
- Robot joint angles  

### Evaluation

- ✅ Task Success Rate  
- ⏱ Completion Time  
- 🔄 Robustness under randomized configurations  
- 🔬 Generalization on unseen scenarios  

---

## 💻 Installation

Make sure you have Python 3.8+ and run:

```bash
pip install pybullet numpy tensorflow keras tqdm
