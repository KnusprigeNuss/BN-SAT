import itertools
import json
import os
from pgmpy.utils import get_example_model
import networkx as nx

def get_pruned_model(model, query_nodes, evidence):
    evidence_keys = list(evidence.keys()) if evidence else []
    targets = list(query_nodes) + evidence_keys
    
    relevant_nodes = set(targets)
    for node in targets:
        relevant_nodes.update(nx.ancestors(model, node))
    
    pruned_model = model.subgraph(relevant_nodes).copy()
    
    for node in relevant_nodes:
        cpd = model.get_cpds(node)
        pruned_model.add_cpds(cpd)
        
    return pruned_model


def generate_wmc_cnf(model_name, query_nodes, evidence=None, out_prefix=None):
    evidence = evidence or {}
    out_prefix = out_prefix or f"temp_res/{model_name}_query"
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    
    # print(f"Building '{model_name}' for Query: {query_nodes} | Evidence: {evidence}")

    model = get_example_model(model_name)
    original_size = len(model.nodes())
    
    model = get_pruned_model(model, query_nodes, evidence)
    # print(f"Pruned Barren Nodes: Reduced from {original_size} to {len(model.nodes())} nodes.")

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
            

    for e_node, e_state in evidence.items():
        if e_node in mapping and e_state in mapping[e_node]:
            e_var = mapping[e_node][e_state]
            clauses.append([e_var])


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
                
                # forward: parents & weight -> child
                clauses.append(parent_lits + [-w_var, child_var])
                # backward: parent & child -> weight
                clauses.append(parent_lits + [-child_var, w_var])
                # weight -> parents
                for p_lit in parent_lits:
                    clauses.append([-w_var, -p_lit])

    num_vars = var_count - 1
    num_clauses = len(clauses)
    
    cnf_file = f"{out_prefix}.cnf"
    wmc_file = f"{out_prefix}.wmc"
    json_file = f"{out_prefix}_data.json"
    
    with open(cnf_file, "w") as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")
            
    with open(wmc_file, "w") as f:
        for var, prob in weights.items():
            f.write(f"w {var} {prob:.15f}\n")
            f.write(f"w -{var} 1.0\n")
            
    data_bundle = {
        "mapping": mapping,
        "weights": weights,
        "num_vars": num_vars,
        "evidence": evidence,
        "query_nodes": query_nodes
    }
    with open(json_file, "w") as f:
        json.dump(data_bundle, f, indent=4)

    return cnf_file, wmc_file, json_file



if __name__ == "__main__":
    generate_wmc_cnf(
        model_name="asia", 
        query_nodes=['tub'], 
        evidence={'asia': 'yes'}
    )