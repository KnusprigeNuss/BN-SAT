import subprocess
import os
import json

def create_wcnf_file(output_name, extra_clauses, m_name):
    with open(f"temp_res/{m_name}.cnf", 'r') as f:
        lines = f.readlines()
    
    num_vars = 0
    cnf_body = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('p cnf'):
            parts = line.split()
            num_vars = int(parts[2])
        elif not line.startswith('c'):
            cnf_body.append(line)
    
    with open(f"temp_res/{m_name}.wmc", 'r') as f:
        weights = {}
        for line in f:
            if line.startswith('w'):
                parts = line.split()
                weights[int(parts[1])] = parts[2]


    # new wcnf file
    num_clauses = len(cnf_body) + len(extra_clauses)
    
    with open(output_name, 'w') as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        
        for line in cnf_body:
            f.write(f"{line}\n")
            
        # evidence list
        for clause in extra_clauses:
            f.write(f"{clause} 0\n")
            
        # weights
        for i in range(1, num_vars + 1):
            w_pos = weights.get(i, "1.0")
            w_neg = weights.get(-i, "1.0")
            f.write(f"c p weight {i} {w_pos} 0\n")
            f.write(f"c p weight -{i} {w_neg} 0\n")



def run_wmc(file_name):
    abs_file_path = os.path.abspath(file_name)
    solver_dir = os.path.dirname(os.path.expanduser("~/sharpsat-td-main/bin/sharpSAT"))
    
    # from sharpsat repo
    cmd = ["./sharpSAT", "-WE",  "-tmpdir", ".", "-prec", "10", abs_file_path]
    # "-decot", "5",
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=solver_dir)

    for line in result.stdout.split('\n'):
        if "exact arb float" in line:
            return float(line.split()[-1])
    return None

def query_probability_node_name(evidence_dict, query_node, query_state, mapping, m_name):
    print(f"Starting query for P({query_node} = {query_state} | {evidence_dict})...")
    
    evidence_vars = []
    for node, state in evidence_dict.items():
        if node in mapping and state in mapping[node]:
            evidence_vars.append(mapping[node][state])
        else:
            print(f"Error: {node}={state} not found in mapping.")
            return None

    if query_node in mapping and query_state in mapping[query_node]:
        query_var = mapping[query_node][query_state]
    else:
        print(f"Error: Query {query_node}={query_state} not found in mapping.")
        return None

    create_wcnf_file("temp_res/temp_denom.wcnf", evidence_vars, m_name)
    denom = run_wmc("temp_res/temp_denom.wcnf")
    # print(f"Denominator result: {denom}")
    
    create_wcnf_file("temp_res/temp_num.wcnf", evidence_vars + [query_var], m_name)
    num = run_wmc("temp_res/temp_num.wcnf")
    # print(f"Numerator result: {num}")
    
    if denom is not None and num is not None:
        if denom == 0: 
            print("Probability is 0 (Impossible evidence).")
            return 0.0
        prob = (num / denom) * 100
        print(f"SUCCESS: P({query_node}={query_state} | {evidence_dict}) = {prob:.2f}%\n")
        return prob
    else:
        print("ERROR: Solver failed to return a result.")
        return None

 
def query_probability_node_int(evidence_list, query_node_var, m_name):
    print(f"Starting query for P({query_node_var} | {evidence_list})...")
    
    create_wcnf_file("temp_res/temp_denom.wcnf", evidence_list, m_name)
    denom = run_wmc("temp_res/temp_denom.wcnf")
    # print(f"Denominator result: {denom}")
    
    create_wcnf_file("temp_res/temp_num.wcnf", evidence_list + [query_node_var], m_name)
    num = run_wmc("temp_res/temp_num.wcnf")
    # print(f"Numerator result: {num}")
    
    if denom is not None and num is not None:
        if denom == 0: return "ERROR"
        prob = (num / denom) * 100
        print(f"SUCCESS: P({query_node_var} | {evidence_list}) = {prob:.2f}%\n")
        return prob
    else:
        print("ERROR: Check solver paths and input file formats.")


if __name__ == "__main__":
    MODEL = "asia"
    with open(f"temp_res/{MODEL}_data.json", "r") as f:
        data = json.load(f)
        mapping = data["mapping"]

    query_probability_node_int([3], 15, MODEL)
    query_probability_node_name({'tub': 'yes'}, 'dysp', 'yes', mapping, MODEL)