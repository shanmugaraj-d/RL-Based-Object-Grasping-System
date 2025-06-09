import pybullet as p
import pybullet_data
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow import keras  # Explicit import
from tqdm import tqdm

# Environment Setup
class RoboticArmEnv:
    def __init__(self):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
        self.object_id = None
        self.reset()

    def reset(self):
        if self.object_id is not None:
            p.removeBody(self.object_id)

        shapes = ['cube', 'sphere', 'cylinder']
        sizes = [0.02, 0.04, 0.06]
        self.shape = random.choice(shapes)
        self.size = random.choice(sizes)

        pos = [random.uniform(0.3, 0.6), random.uniform(-0.2, 0.2), self.size]

        if self.shape == 'cube':
            self.object_id = p.loadURDF("cube_small.urdf", pos, globalScaling=(self.size / 0.02))
        elif self.shape == 'sphere':
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=self.size, rgbaColor=[1, 0, 0, 1])
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.size)
            self.object_id = p.createMultiBody(0.1, collision_shape, visual_shape, pos)
        elif self.shape == 'cylinder':
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=self.size, length=0.1, rgbaColor=[0, 1, 0, 1])
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.size, height=0.1)
            self.object_id = p.createMultiBody(0.1, collision_shape, visual_shape, pos)

        p.stepSimulation()
        time.sleep(0.1)
        return self.get_state()

    def get_state(self):
        object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        joint_states = [p.getJointState(self.robot_id, i)[0] for i in range(p.getNumJoints(self.robot_id))]
        dist = np.linalg.norm(np.array(object_pos[:2]) - np.array([0, 0]))
        shape_encoding = [int(self.shape == s) for s in ['cube', 'sphere', 'cylinder']]
        return np.array([dist] + shape_encoding + [self.size] + joint_states)

    def step(self, action):
        for i in range(min(len(action), p.getNumJoints(self.robot_id))):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=action[i])
        p.stepSimulation()
        time.sleep(0.01)

        state = self.get_state()
        reward = -state[0]
        done = state[0] < 0.05
        if done:
            reward = 10
        return state, reward, done

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(state_size,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(action_size, activation='linear')
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    def predict(self, state):
        return self.model.predict(np.array([state]), verbose=0)[0]

    def train(self, state, target_q):
        self.model.fit(np.array([state]), np.array([target_q]), verbose=0)

# Train the model
env = RoboticArmEnv()
state_size = len(env.get_state())
action_size = 7  # Number of joints
agent = DQNAgent(state_size, action_size)

episodes = 10  # Reduce for faster testing
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.98
epsilon_min = 0.1

print("🚀 Starting training with object shape & size awareness...")

for e in tqdm(range(episodes), desc="Training Progress"):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done and steps < 50:
        if random.random() < epsilon:
            action = np.random.uniform(-1, 1, action_size)
        else:
            action = agent.predict(state)

        next_state, reward, done = env.step(action)
        target_q = agent.predict(state)
        future_q = agent.predict(next_state)
        target_q[np.argmax(target_q)] = reward + gamma * np.max(future_q)
        agent.train(state, target_q)

        state = next_state
        total_reward += reward
        steps += 1

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"🏁 Episode {e+1} | Reward: {total_reward:.2f} | Steps: {steps} | Object: {env.shape} | Size: {env.size:.2f}")

print("\n✅ Training complete! Press Ctrl+C in the terminal to stop the simulation.")

# Keep simulation window open
try:
    while True:
        p.stepSimulation()
        time.sleep(0.01)
except KeyboardInterrupt:
    print("👋 Simulation ended by user.")
    p.disconnect()
