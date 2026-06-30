import subprocess
import os
import json
import resource

GANAK_PATH = os.path.abspath("./solvers_bin/ganak") 
BASE_DIR = "temp_res"

def limit_memory():
    try:
        limit = 8 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except ValueError:
        pass


def create_wcnf_file(output_name, extra_clauses, m_name):
    with open(f"{BASE_DIR}/{m_name}.cnf", 'r') as f:
        lines = f.readlines()
    
    num_vars = 0
    cnf_body = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('c'):
            continue
            
        if line.startswith('p cnf'):
            parts = line.split()
            num_vars = int(parts[2])
        else:
            cnf_body.append(line)
    
    with open(f"{BASE_DIR}/{m_name}.wmc", 'r') as f:
        weights = {}
        for line in f:
            if line.startswith('w'):
                parts = line.split()
                weights[int(parts[1])] = parts[2]

    num_clauses = len(cnf_body) + len(extra_clauses)
    
    with open(output_name, 'w') as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        
        for line in cnf_body:
            f.write(f"{line}\n")
            
        for clause in extra_clauses:
            f.write(f"{clause} 0\n")
            
        for i in range(1, num_vars + 1):
            w_pos = weights.get(i, "1.0")
            w_neg = weights.get(-i, "1.0")
            f.write(f"c p weight {i} {w_pos} 0\n")
            f.write(f"c p weight -{i} {w_neg} 0\n")


def run_wmc(file_name):
    abs_file_path = os.path.abspath(file_name)
    
    cmd = [GANAK_PATH, "--mode", "1", abs_file_path]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, preexec_fn=limit_memory)

    for line in result.stdout.split('\n'):
        if "exact quadruple float" in line:
            return float(line.split()[-1])
    return None


def query_probability_ganak(evidence_dict, query_node, query_state, mapping, m_name):
    # print(f"Ganak Query: P({query_node}={query_state} | {evidence_dict})")
    
    evidence_vars = [mapping[node][state] for node, state in evidence_dict.items()]
    query_var = mapping[query_node][query_state]

    create_wcnf_file(f"{BASE_DIR}/ganak_denom.cnf", evidence_vars, m_name)
    denom = run_wmc(f"{BASE_DIR}/ganak_denom.cnf")
    
    create_wcnf_file(f"{BASE_DIR}/ganak_num.cnf", evidence_vars + [query_var], m_name)
    num = run_wmc(f"{BASE_DIR}/ganak_num.cnf")
    
    if denom is not None and num is not None:
        if denom == 0: return 0.0
        prob = (num / denom) * 100
        # print(f"Ganak Result: {prob:.2f}%\n")
        return prob
    return None


def get_joint_wmc_ganak(assignment_dict, mapping, m_name):
    assignment_vars = []
    for node, state in assignment_dict.items():
        if node in mapping and state in mapping[node]:
            assignment_vars.append(mapping[node][state])
        else:
            print(f"Error: {node}={state} not found in mapping.")
            return 0.0

    temp_file = f"{BASE_DIR}/temp_map_ganak.cnf"
    create_wcnf_file(temp_file, assignment_vars, m_name)
    weight = run_wmc(temp_file)
    
    return weight if weight is not None else 0.0


if __name__ == "__main__":
    MODEL = "alarm"
    
    with open(f"temp_res/{MODEL}_data.json", "r") as f:
        data = json.load(f)
        mapping = data["mapping"]

    query_probability_ganak({'MINVOLSET': 'NORMAL'}, 'CO', 'LOW', mapping, MODEL)