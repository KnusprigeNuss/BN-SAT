import time
import random
import sys
import numpy as np
from pgmpy.utils import get_example_model
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from bn_to_mpe_sat import generate_mpe_encoding

sys.setrecursionlimit(20000)

MODELS = ["asia", "alarm", "hailfinder"]
EVIDENCE = {
    'asia': {'tub': 'yes', 'dysp': 'no'}, 
    'alarm': {'MINVOL': 'NORMAL'},
    'hailfinder': {'TempDis': 'None', 'CombClouds': 'PC'},
    'hepar2' : {}
}

def get_fast_joint_prob(model, state):
    prob = 1.0
    for cpd in model.get_cpds():
        var = cpd.variable
        var_idx = cpd.name_to_no[var][state[var]]
        
        if not cpd.variables[1:]:
            p = cpd.values[var_idx]
        else:
            parent_vars = cpd.variables[1:]
            parent_indices = tuple(cpd.name_to_no[p][state[p]] for p in parent_vars)
            idx = (var_idx,) + parent_indices
            p = cpd.values[idx]
            
        prob *= p
        if prob == 0.0:
            return 0.0
    return prob

def classical_hill_climbing(model, evidence, max_iterations=5000):
    nodes = model.nodes()
    mpe_nodes = [n for n in nodes if n not in evidence]
    
    current_state = {**evidence}
    for n in mpe_nodes:
        states = model.get_cpds(n).state_names[n]
        current_state[n] = random.choice(states)
        
    current_prob = get_fast_joint_prob(model, current_state)

    # loop: pick random node and flip state, better -> keep
    for _ in range(max_iterations):
        flip_node = random.choice(mpe_nodes)
        states = model.get_cpds(flip_node).state_names[flip_node]
        
        old_val = current_state[flip_node]
        new_val = random.choice([s for s in states if s != old_val])
        
        current_state[flip_node] = new_val
        new_prob = get_fast_joint_prob(model, current_state)
        
        if new_prob > current_prob:
            current_prob = new_prob
        else:
            current_state[flip_node] = old_val
            
    return current_prob, current_state


def get_assignment_from_model(sat_model, mapping):
    assignment = {}
    for node, states in mapping.items():
        for state, sat_var in states.items():
            if sat_var in sat_model:
                assignment[node] = state
    return assignment


def run_hill_climbing_tournament():
    for m_name in MODELS:
        print(f"\nMpe Tournament: Hillclimb vs SAT - {m_name.upper()}")
        model = get_example_model(m_name)
        mpe_nodes = [n for n in model.nodes() if n not in EVIDENCE[m_name]]
        
        print("start rc2")
        mapping_rc2, _ = generate_mpe_encoding(model, EVIDENCE[m_name], m_name)
        wcnf_file = f"temp_res/{m_name}_mpe.wcnf"
        
        start_rc2 = time.perf_counter()
        wcnf = WCNF(from_file=wcnf_file)
        with RC2(wcnf) as rc2:
            rc2_model = rc2.compute()
            rc2_time = time.perf_counter() - start_rc2
            rc2_assignment = get_assignment_from_model(rc2_model, mapping_rc2)
            
        rc2_prob = get_fast_joint_prob(model, rc2_assignment)
        print("end rc2")


        print("start hc")
        start_hc = time.perf_counter()
        hc_prob, hc_assignment = classical_hill_climbing(model, EVIDENCE[m_name], max_iterations=10000)
        hc_time = time.perf_counter() - start_hc
        print("end hc")


        # --- 3. Validation ---
        match = all(rc2_assignment.get(n) == hc_assignment.get(n) for n in mpe_nodes)
        # rc2_time = 0
        # rc2_prob = 0
        # match = False

        print("-" * 80)
        print(f"{'Solver Paradigm':<25} | {'Time (s)':<12} | {'Max Prob':<20} | {'MPE Match?'}")
        print("-" * 80)
        print(f"{'RC2 (Exact SAT)':<25} | {rc2_time:<12.4f} | {rc2_prob:<20.6e} | Base Truth")
        print(f"{'Hill Climbing (Classical)':<25} | {hc_time:<12.4f} | {hc_prob:<20.6e} | {'YES' if match else 'NO'}")
        print("-" * 80)

if __name__ == "__main__":
    run_hill_climbing_tournament()