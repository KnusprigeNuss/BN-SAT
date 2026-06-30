import itertools
from pgmpy.utils import get_example_model
import json
import os
import networkx as nx

def prune_barren_nodes(model, map_nodes, evidence):
    essential_nodes = set(map_nodes).union(set(evidence.keys()))
    
    pruned = True
    while pruned:
        pruned = False
        leaves = [node for node in model.nodes() if model.out_degree(node) == 0]
        for leaf in leaves:
            if leaf not in essential_nodes:
                model.remove_node(leaf)
                pruned = True
    return model

def generate_map_cnf(model_name, map_nodes, evidence=None, out_prefix=None):
    evidence = evidence or {}
    out_prefix = out_prefix or f"temp_res/{model_name}_map"
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    
    # print(f"Generator Building {model_name} with Evidence: {evidence}")

    model = get_example_model(model_name)

    # here happens the pruning, can be deactivated if needed ########
    original_size = len(model.nodes())
    model = prune_barren_nodes(model, map_nodes, evidence)
    # print(f"Pruned Barren Nodes: Reduced from {original_size} to {len(model.nodes())} nodes.")
    #########################################################

    mapping = {} 
    clauses = []
    weights = {} 
    var_count = 1
    weight_context = {}

    # nodes and states
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

    # evidence
    for e_node, e_state in evidence.items():
        if e_node in mapping and e_state in mapping[e_node]:
            e_var = mapping[e_node][e_state]
            clauses.append([e_var]) 

    # cpds
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
                
                if prob == 0.0:
                    clauses.append(parent_lits + [-mapping[node][state_name]])
                    continue
                
                w_var = var_count
                var_count += 1
                weights[w_var] = prob
                
                child_var = mapping[node][state_name]
                
                parent_ctx = ", ".join([f"{p}({s})" for p, s in zip(parents, combination)])
                node_info = f"{node}({state_name})"
                parent_info = parent_ctx if parents else "None"
                weight_context[w_var] = (node_info, parent_info)
                
                # forward: parents & weight -> child
                clauses.append(parent_lits + [-w_var, child_var])
                
                # backward: parent & child -> weight
                clauses.append(parent_lits + [-child_var, w_var])

                # weight -> parents
                for p_lit in parent_lits:
                    clauses.append([-w_var, -p_lit])

    num_vars = var_count - 1
    
    # map nodes
    map_sat_vars = []
    for node in map_nodes:
        if node in mapping:
            map_sat_vars.extend(mapping[node].values())
            
    # print(f"Projecting onto {len(map_sat_vars)} target variables.")

    # cnf
    cnf_file = f"{out_prefix}.cnf"
    with open(cnf_file, "w") as f:
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
        "evidence": evidence,
        "map_nodes": map_nodes,
        "clauses": clauses
    }
    
    json_file = f"{out_prefix}_data.json"
    with open(json_file, "w") as f:
        json.dump(data_bundle, f, indent=4)
        
    return cnf_file, json_file



if __name__ == "__main__":
    generate_map_cnf(
        model_name="asia", 
        map_nodes=['lung', 'bronc'], 
        evidence={'xray': 'yes'}
    )