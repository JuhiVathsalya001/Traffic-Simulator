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
