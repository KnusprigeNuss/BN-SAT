import json

def get_weight(lit, evidence_lits, cpt_weights):
    var = abs(lit)
    if -lit in evidence_lits:
        return 0.0
    if lit in evidence_lits:
        return 1.0
    
    if var in cpt_weights:
        return cpt_weights[var] if lit > 0 else 1.0
    
    return 1.0 

def solve_circuit(nnf_lines, evidence_nodes, mapping, cpt_weights):
    evidence_lits = set()
    for node, state in evidence_nodes.items():
        evidence_lits.add(mapping[node][state])

    nodes = {} 
    edges = [] 
    
    for line in nnf_lines:
        parts = line.split()
        if not parts: continue
        if parts[0] in ['o', 'a', 't', 'f']:
            nodes[int(parts[1])] = {'type': parts[0], 'val': None}
        else:
            parent, child = int(parts[0]), int(parts[1])
            lits = [int(x) for x in parts[2:-1]]
            edges.append((parent, child, lits))

    memo = {}

    def compute(node_id, cpt_weights):
        if node_id in memo: return memo[node_id]
        
        node = nodes[node_id]
        if node['type'] == 't': return 1.0
        if node['type'] == 'f': return 0.0 
        
        child_edges = [e for e in edges if e[0] == node_id]
        
        if node['type'] == 'a':
            res = 1.0
            for _, child, lits in child_edges:
                branch_val = compute(child, cpt_weights)
                for l in lits: branch_val *= get_weight(l, evidence_lits, cpt_weights)
                res *= branch_val
            memo[node_id] = res
            return res
            
        if node['type'] == 'o':
            res = 0.0
            for _, child, lits in child_edges:
                branch_val = compute(child, cpt_weights)
                for l in lits: branch_val *= get_weight(l, evidence_lits, cpt_weights)
                res += branch_val
            memo[node_id] = res
            return res

    return compute(1, cpt_weights) 


if __name__ == "__main__":
    with open("temp_res/model_data.json", "r") as f:
        data = json.load(f)
        mapping = data["mapping"]
        cpt_weights = {int(k): v for k, v in data["weights"].items()}
        
    with open("temp_res/asia.nnf", "r") as f:
        nnf_lines = f.readlines()

    ev_denom = {'tub': 'yes'}
    ev_num = {'tub': 'yes', 'lung': 'yes'}

    p_e = solve_circuit(nnf_lines, ev_denom, mapping, cpt_weights)
    p_qe = solve_circuit(nnf_lines, ev_num, mapping, cpt_weights)

    print(f"Denominator P(tub=yes): {p_e}")
    print(f"Numerator P(tub=yes, lung=yes): {p_qe}")
    print(f"Conditional P(lung=yes | tub=yes): {p_qe / p_e if p_e > 0 else 0}")