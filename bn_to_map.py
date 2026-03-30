import itertools
from pgmpy.utils import get_example_model
import json

MODEL = "asia"
model = get_example_model(MODEL)
mapping = {} 
clauses = []
weights = {} 
var_count = 1
weight_context = {}

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
    # print(node_states)

    parent_state_lists = [cpd.state_names[p] for p in parents]
    # print(parent_state_lists)
    state_combinations = list(itertools.product(*parent_state_lists))
    # print(node)
    # print(state_combinations)


    for combination in state_combinations:
        # combinations of parents and states with their SAT var. 
        parent_lits = [-mapping[p][s] for p, s in zip(parents, combination)]
        
        parent_state_indices = [cpd.get_state_no(p, s) for p, s in zip(parents, combination)]
        
        for state_idx, state_name in enumerate(node_states):
            # fetch cpt value
            prob = float(cpd.values[tuple([state_idx] + parent_state_indices)])
            
            # optimization?
            # if prob == 0:
            #     clauses.append(parent_lits + [-mapping[node][state_name]])
            #     continue
            
            w_var = var_count
            var_count += 1
            weights[w_var] = prob
            
            child_var = mapping[node][state_name]
            
            # for debugging
            parent_ctx = ", ".join([f"{p}({s})" for p, s in zip(parents, combination)])
            node_info = f"{node}({state_name})"
            parent_info = parent_ctx if parents else "None"
            weight_context[w_var] = (node_info, parent_info)
            
            # forward: parents & weight -> child
            clauses.append(parent_lits + [-w_var, child_var])
            
            # backward: parent & child -> weight
            clauses.append(parent_lits + [-child_var, w_var])

            # weight -> parents
            # weight must be false if any parent is not in correct state according to CPT
            for p_lit in parent_lits:
                clauses.append([-w_var, -p_lit])

def save_for_map(clauses, weights, filename, target_nodes=[]):
    num_vars = var_count - 1
    
    map_sat_vars = []
    for node in target_nodes:
        if node in mapping:
            map_sat_vars.extend(mapping[node].values())
    
    weight_vars = list(weights.keys())
    map_sat_vars.extend(weight_vars)
            
    print(f"Projecting onto {len(map_sat_vars)} variables (Targets + Weights)")

    with open(f"temp_res/{filename}_map.cnf", "w") as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        
        if map_sat_vars:
            f.write("c p show " + " ".join(map(str, map_sat_vars)) + " 0\n")
            
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")
        
        for var, prob in weights.items():
            f.write(f"c p weight {var} {prob:.15f} 0\n")
            f.write(f"c p weight -{var} 1.0 0\n")


    data_bundle = {
        "mapping": mapping,
        "weights": weights,
        "num_vars": num_vars,
        "clauses": clauses
    }
    
    with open("temp_res/model_data.json", "w") as f:
        json.dump(data_bundle, f)

save_for_map(clauses, weights, MODEL, target_nodes=['lung', 'bronc'])