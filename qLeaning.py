import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    tools=os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable SUMO_HOME")

import traci

#configuration need to be chnaged
Sumo_config = [
    'sumo-gui',
    '-c', 'RL.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
]
#

traci.start(Sumo_config)
traci.gui.setSchema("View #0","real-world")

#detectors-to be chnaged
q_EB_0 = 0
q_EB_1 = 0
q_EB_2 = 0
q_SB_0 = 0
q_SB_1 = 0
q_SB_2 = 0
current_phase = 0
#

TOTAL_STEPS=1000
ALPHA=0.1
GAMMA=0.9
EPSILON=0.0
ACTIONS=[0,1]

Q_table={}

MIN_GREEN_STEPS=0
last_switch_step=-MIN_GREEN_STEPS

def get_max_Q_value_of_state(s):
    if s not in Q_table:
        Q_table[s]=np.zeros(len(ACTIONS))
    return np.max(Q_table[s])

def get_reward(state):
    total_queue=sum(state[:-1])
    reward=-float(total_queue)
    return reward

#might need changes here
def get_state():  
    global q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase
    
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"
    
    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"
    
  
    traffic_light_id = "Node2"
    
  
    q_EB_0 = get_queue_length(detector_Node1_2_EB_0)
    q_EB_1 = get_queue_length(detector_Node1_2_EB_1)
    q_EB_2 = get_queue_length(detector_Node1_2_EB_2)
    
    q_SB_0 = get_queue_length(detector_Node2_7_SB_0)
    q_SB_1 = get_queue_length(detector_Node2_7_SB_1)
    q_SB_2 = get_queue_length(detector_Node2_7_SB_2)
    
    current_phase = get_current_phase(traffic_light_id)
    
    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)

def apply_action(action, tls_id="Node2"): 
    """
    Executes the chosen action on the traffic light, combining:
      - Min Green Time check
      - Switching to the next phase if allowed
    Constraint #5: Ensure at least MIN_GREEN_STEPS pass before switching again.
    """
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
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))
    
    
    # 1) Predict current Q-values from old_state (current state)
    old_q = Q_table[old_state][action]
    # 2) Predict Q-values for new_state to get max future Q (new state)
    best_future_q = get_max_Q_value_of_state(new_state)
    # 3) Incorporate ALPHA to partially update the Q-value and update Q table
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)







def get_action_from_policy(state): 
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        return 0




def get_queue_length(detector_id): 
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id): 
    return traci.trafficlight.getPhase(tls_id)


# Lists to record data for plotting
step_history = []
reward_history = []
queue_history = []

cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step
    
    state = get_state()
    #action = get_action_from_policy(state)
    #apply_action(action)
    
    traci.simulationStep()  
    
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward
    
    #update_Q_table(state, action, reward, new_state)
    
    # Print Q-values for the old_state right after update
    #updated_q_vals = Q_table[state]

    if step % 100 == 0:
        print(f"Step {step}, Current_State: {state}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))   
        print("Current Q-table:")
        for st, qvals in Q_table.items():
            print(f"  {st} -> {qvals}")


traci.close()

print("\nOnline Training completed. Final Q-table size:", len(Q_table))
for st, actions in Q_table.items():
    print("State:", st, "-> Q-values:", actions)


plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("Fixed Timing: Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("Fixed Timing: Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()
