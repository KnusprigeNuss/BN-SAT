import math
import itertools
from pgmpy.utils import get_example_model

from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
import os

mapping = {} 
clauses = []
weights = {} 
var_count = 1

SCALE = 1000000 
HARD_WEIGHT = 10**15

def generate_mpe_encoding(model, evidence, filename):
    mapping = {}
    clauses = []
    weights = {}
    var_count = 1

    for node in model.nodes():
        cpd = model.get_cpds(node)
        states = cpd.state_names[node]
        mapping[node] = {}
        state_vars = []
        for state in states:
            mapping[node][state] = var_count
            state_vars.append(var_count)
            var_count += 1
        
        clauses.append(state_vars)
        for pair in itertools.combinations(state_vars, 2):
            clauses.append([-pair[0], -pair[1]])

    for cpd in model.get_cpds():
        node = cpd.variable
        parents = cpd.variables[1:]
        node_states = cpd.state_names[node]
        parent_state_lists = [cpd.state_names[p] for p in parents]
        state_combinations = list(itertools.product(*parent_state_lists))

        for combination in state_combinations:
            parent_lits = [-mapping[p][s] for p, s in zip(parents, combination)]
            parent_state_indices = [cpd.get_state_no(p, s) for p, s in zip(parents, combination)]
            
            for state_idx, state_name in enumerate(node_states):
                prob = float(cpd.values[tuple([state_idx] + parent_state_indices)])
                
                if prob == 0:
                    clauses.append(parent_lits + [-mapping[node][state_name]])
                    continue

                w_var = var_count
                var_count += 1
                
                # convert probability to logarithmic cost for Max-SAT 
                weights[w_var] = int(-math.log10(prob) * SCALE)
                child_var = mapping[node][state_name]
                
                clauses.append(parent_lits + [-w_var, child_var])
                clauses.append(parent_lits + [-child_var, w_var])
                for p_lit in parent_lits:
                    clauses.append([-w_var, -p_lit])


    for node, state in evidence.items():
        if node in mapping and state in mapping[node]:
            clauses.append([mapping[node][state]])


    output_path = f"temp_res/{filename}_mpe.wcnf"
    num_vars = var_count - 1
    num_clauses = len(clauses) + len(weights)
    
    with open(output_path, "w") as f:
        f.write(f"p wcnf {num_vars} {num_clauses} {HARD_WEIGHT}\n")
        for c in clauses:
            f.write(f"{HARD_WEIGHT} " + " ".join(map(str, c)) + " 0\n")
        for var, cost in weights.items():
            f.write(f"{cost} -{var} 0\n")

    return mapping, weights



def save_for_mpe(clauses, weights, filename):
    num_vars = var_count - 1
    num_clauses = len(clauses) + len(weights)
    
    output_path = f"temp_res/{filename}_mpe.wcnf"
    with open(output_path, "w") as f:
        f.write(f"p wcnf {num_vars} {num_clauses} {HARD_WEIGHT}\n")
        
        for clause in clauses:
            f.write(f"{HARD_WEIGHT} " + " ".join(map(str, clause)) + " 0\n")
            
        for var, cost in weights.items():
            f.write(f"{cost:.6f} {var} 0\n")


def save_mpe_wcnf(filename):
    with open(f"temp_res/{filename}_mpe.wcnf", "w") as f:
        f.write(f"p wcnf {var_count-1} {len(clauses) + len(weights)} {HARD_WEIGHT}\n")
        for c in clauses:
            f.write(f"{HARD_WEIGHT} " + " ".join(map(str, c)) + " 0\n")
        for var, cost in weights.items():
            f.write(f"{cost} -{var} 0\n")


def solve_mpe(wcnf_file, mapping):
    wcnf = WCNF(from_file=wcnf_file)
    with RC2(wcnf) as rc2:
        model = rc2.compute() 
        
        if model:
            print("\nMost Probable Explanation Found:")
            print("-" * 40)
            for node, states in mapping.items():
                for state, sat_var in states.items():
                    if sat_var in model:
                        print(f"{node:<15}: {state}")
        else:
            print("No solution found (Hard clauses unsatisfied).")

if __name__ == "__main__":
    MODEL = "alarm"
    model = get_example_model(MODEL)
    for node in model.nodes():
        cpd = model.get_cpds(node)
        states = cpd.state_names[node]
        mapping[node] = {}
        state_vars = []
        for state in states:
            mapping[node][state] = var_count
            state_vars.append(var_count)
            var_count += 1
        
        clauses.append(state_vars)
        for pair in itertools.combinations(state_vars, 2):
            clauses.append([-pair[0], -pair[1]])

    for cpd in model.get_cpds():
        node = cpd.variable
        parents = cpd.variables[1:]
        node_states = cpd.state_names[node]
        parent_state_lists = [cpd.state_names[p] for p in parents]
        state_combinations = list(itertools.product(*parent_state_lists))

        for combination in state_combinations:
            parent_lits = [-mapping[p][s] for p, s in zip(parents, combination)]
            parent_state_indices = [cpd.get_state_no(p, s) for p, s in zip(parents, combination)]
            
            for state_idx, state_name in enumerate(node_states):
                prob = float(cpd.values[tuple([state_idx] + parent_state_indices)])
                
                if prob == 0:
                    clauses.append(parent_lits + [-mapping[node][state_name]])
                    continue

                w_var = var_count
                var_count += 1
                
                weights[w_var] = int(-math.log10(prob) * SCALE)
                
                child_var = mapping[node][state_name]
                
                clauses.append(parent_lits + [-w_var, child_var])
                clauses.append(parent_lits + [-child_var, w_var])
                for p_lit in parent_lits:
                    clauses.append([-w_var, -p_lit])

    # evidence = {'either': 'yes'} #dysp':'yes' 
    evidence = {'MINVOL': 'NORMAL'} 
    for node, state in evidence.items():
        if node in mapping and state in mapping[node]:
            clauses.append([mapping[node][state]])
        else:
            print(f"Warning: Evidence {node}={state} not found in mapping.")

    # mapping, weights = generate_mpe_encoding(model, evidence, MODEL)

    save_mpe_wcnf(MODEL)
    solve_mpe(f"temp_res/{MODEL}_mpe.wcnf", mapping)