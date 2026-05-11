import subprocess
import itertools
import os
from pgmpy.utils import get_example_model
import json

MODEL_NAME = "asia"
GANAK_PATH = "./solvers_bin/ganak" 
TARGET_NODES = ['lung', 'bronc']
EVIDENCE = {'tub': 'yes'}

with open(f"temp_res/{MODEL_NAME}_data.json", "r") as f:
    data = json.load(f)
    mapping = data["mapping"]
    weights = data["weights"]
    clauses = data["clauses"]
    num_vars = data["num_vars"]

model = get_example_model(MODEL_NAME)


def run_ganak_query(extra_clauses):
    temp_file = "temp_res/map_query.cnf"
    
    with open(temp_file, "w") as f:
        f.write(f"p cnf {num_vars} {len(clauses) + len(extra_clauses)}\n")
        
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")
        for c in extra_clauses:
            f.write(f"{c} 0\n")
            
        for var, prob in weights.items():
            f.write(f"cp weight {var} {prob:.15f} 0\n")
            f.write(f"cp weight -{var} 1.000000 0\n")

    cmd = [GANAK_PATH, "--mode", "1", temp_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    for line in result.stdout.split('\n'):
        if "exact quadruple float" in line:
            return float(line.split()[-1])
    return 0.0

print(f"--- Solving MAP for {TARGET_NODES} given {EVIDENCE} ---")
best_prob = -1
best_config = None

target_states = [model.get_cpds(n).state_names[n] for n in TARGET_NODES]
combinations = list(itertools.product(*target_states))

for combo in combinations:
    current_evidence_lits = []
    
    for node, state in zip(TARGET_NODES, combo):
        current_evidence_lits.append(mapping[node][state])
    
    for node, state in EVIDENCE.items():
        current_evidence_lits.append(mapping[node][state])
        
    prob = run_ganak_query(current_evidence_lits)
    print(f"Config {dict(zip(TARGET_NODES, combo))}: Prob = {prob:.6f}")
    
    if prob > best_prob:
        best_prob = prob
        best_config = combo

print("-" * 50)
print(f"MAP Result: {dict(zip(TARGET_NODES, best_config))}")
print(f"Joint Probability P(M, e): {best_prob:.6f}")