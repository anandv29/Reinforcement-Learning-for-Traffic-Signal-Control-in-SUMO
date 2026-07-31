# =============================================================
# Deep Q-Learning + SUMO (Stable TensorFlow Integration)
# =============================================================

# Step 0: Safe TensorFlow import order (critical on Windows)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"           # Hide TF info/warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # Prevent GPU memory pre-allocation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # Force CPU (avoids DLL GPU issues)

# Step 1.1: (Additional) Imports for Deep Q-Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Step 1: Add modules to provide access to specific libraries and functions
import sys
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt  # Visualization

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to SUMO
import traci  # Static network information

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMO_CONFIG = PROJECT_ROOT / "sumo" / "RL.sumocfg"

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo',                     # Use CLI (stable). Change to 'sumo-gui' if you need GUI.
    '-c', str(SUMO_CONFIG),
    '--step-length', '0.10',
    '--lateral-resolution', '0'
]

# Step 5: Open connection between SUMO and TraCI
traci.start(Sumo_config)
print("✅ SUMO simulation connected successfully")

# -------------------------
# Step 6: Define Variables
# -------------------------

q_EB_0 = q_EB_1 = q_EB_2 = 0
q_SB_0 = q_SB_1 = q_SB_2 = 0
current_phase = 0

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 10000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
ACTIONS = [0, 1]  # (0 = keep phase, 1 = switch phase)

# ---- Stability ----
MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# Step 7: Define Functions
# -------------------------

def build_model(state_size, action_size):
    """Build a simple feedforward neural network that approximates Q-values."""
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    return model

def to_array(state_tuple):
    """Convert the state tuple into a NumPy array for neural network input."""
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))

# Create DQN model
state_size = 7
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)

def get_max_Q_value_of_state(s):
    state_array = to_array(s)
    Q_values = dqn_model.predict(state_array, verbose=0)[0]
    return np.max(Q_values)

def get_reward(state):
    """Negative of total queue length to encourage shorter queues."""
    total_queue = sum(state[:-1])
    reward = -float(total_queue)
    return reward

def get_state():
    global q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase
    
    # Detector IDs
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"
    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"
    
    # Traffic light ID
    traffic_light_id = "Nod"
    
    # Queue lengths
    q_EB_0 = get_queue_length(detector_Node1_2_EB_0)
    q_EB_1 = get_queue_length(detector_Node1_2_EB_1)
    q_EB_2 = get_queue_length(detector_Node1_2_EB_2)
    q_SB_0 = get_queue_length(detector_Node2_7_SB_0)
    q_SB_1 = get_queue_length(detector_Node2_7_SB_1)
    q_SB_2 = get_queue_length(detector_Node2_7_SB_2)
    
    # Current phase
    current_phase = get_current_phase(traffic_light_id)
    
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)

def apply_action(action, tls_id="Nod"):
    """Executes the chosen action (switch or hold)."""
    global last_switch_step
    
    if action == 0:
        return
    elif action == 1:
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_simulation_step

def update_Q_table(old_state, action, reward, new_state):
    """Perform DQN update."""
    old_state_array = to_array(old_state)
    new_state_array = to_array(new_state)

    Q_values_old = dqn_model.predict(old_state_array, verbose=0)[0]
    Q_values_new = dqn_model.predict(new_state_array, verbose=0)[0]
    best_future_q = np.max(Q_values_new)
    
    Q_values_old[action] += ALPHA * (reward + GAMMA * best_future_q - Q_values_old[action])
    dqn_model.fit(old_state_array, np.array([Q_values_old]), verbose=0)

def get_action_from_policy(state):
    """Epsilon-greedy strategy using the DQN's predicted Q-values."""
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        state_array = to_array(state)
        Q_values = dqn_model.predict(state_array, verbose=0)[0]
        return int(np.argmax(Q_values))

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

# -------------------------
# Step 8: Training Loop
# -------------------------
step_history, reward_history, queue_history = [], [], []
cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")

for step in range(TOTAL_STEPS):
    current_simulation_step = step
    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)
    
    traci.simulationStep()
    
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward
    
    update_Q_table(state, action, reward, new_state)
    
    updated_q_vals = dqn_model.predict(to_array(state), verbose=0)[0]
    print(f"Step {step}, State: {state}, Action: {action}, Reward: {reward:.2f}, "
          f"Cumulative: {cumulative_reward:.2f}, Q: {updated_q_vals}")
    
    step_history.append(step)
    reward_history.append(cumulative_reward)
    queue_history.append(sum(new_state[:-1]))

# -------------------------
# Step 9: Close TraCI + Results
# -------------------------
traci.close()
print("\n✅ Online Training completed.")
dqn_model.summary()

plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("DQN Training: Cumulative Reward")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("DQN Training: Queue Length")
plt.grid(True)
plt.legend()
plt.show()
