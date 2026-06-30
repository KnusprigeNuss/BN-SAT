import subprocess
import os
import json
import resource

def limit_memory():
    try:
        limit = 8 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except ValueError:
        pass


def create_wcnf_file(output_name, extra_clauses, m_name):
    cnf_path = f"temp_res/{m_name}.cnf"
    if not os.path.exists(cnf_path):
        print(f"ERROR: {cnf_path} not found.")
        return False
        
    with open(cnf_path, 'r') as f:
        lines = f.readlines()
    
    num_vars = 0
    cnf_body = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('c'): continue
        if line.startswith('p cnf'):
            num_vars = int(line.split()[2])
        else:
            cnf_body.append(line)
    
    weights = {}
    wmc_path = f"temp_res/{m_name}.wmc"
    # maybe better like that
    # json_path = f"temp_res/{m_name}_data.json"
    json_path = "temp_res/model_data.json"

    
    if os.path.exists(wmc_path):
        with open(wmc_path, 'r') as f:
            for line in f:
                if line.startswith('w'):
                    parts = line.split()
                    weights[int(parts[1])] = parts[2]
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            weights = {int(k): str(v) for k, v in data.get("weights", {}).items()}

    num_clauses = len(cnf_body) + len(extra_clauses)
    with open(output_name, 'w') as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for line in cnf_body: f.write(f"{line}\n")
        for var in extra_clauses: f.write(f"{var} 0\n")
        for i in range(1, num_vars + 1):
            w_pos = weights.get(i, "1.0")
            w_neg = weights.get(-i, "1.0")
            f.write(f"c p weight {i} {w_pos} 0\n")
            f.write(f"c p weight -{i} {w_neg} 0\n")
    return True


def run_wmc(file_name):
    abs_file_path = os.path.abspath(file_name)
    solver_dir = os.path.dirname(os.path.expanduser("~/sharpsat-td-main/bin/sharpSAT"))

    cmd = ["./sharpSAT", "-WE", "-decot", "0.001",  "-tmpdir", ".", "-prec", "10", abs_file_path]
    # "-decot", "5",
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=solver_dir, timeout=300, preexec_fn=limit_memory)

    for line in result.stdout.split('\n'):
        if "exact arb float" in line:
            res = float(line.split()[-1])
            return res
    return None


def query_probability_node_name(evidence_dict, query_node, query_state, mapping, m_name):
    # print(f"Starting query for P({query_node} = {query_state} | {evidence_dict})...")
    
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
        # print(f"SUCCESS: P({query_node}={query_state} | {evidence_dict}) = {prob:.2f}%\n")
        return prob
    else:
        print("ERROR: Solver failed to return a result.")
        return None

 
def query_probability_node_int(evidence_list, query_node_var, m_name):
    # print(f"Starting query for P({query_node_var} | {evidence_list})...")
    
    create_wcnf_file("temp_res/temp_denom.wcnf", evidence_list, m_name)
    denom = run_wmc("temp_res/temp_denom.wcnf")
    # print(f"Denominator result: {denom}")
    
    create_wcnf_file("temp_res/temp_num.wcnf", evidence_list + [query_node_var], m_name)
    num = run_wmc("temp_res/temp_num.wcnf")
    # print(f"Numerator result: {num}")
    
    if denom is not None and num is not None:
        if denom == 0: return "ERROR"
        prob = (num / denom) * 100
        # print(f"SUCCESS: P({query_node_var} | {evidence_list}) = {prob:.2f}%\n")
        return prob
    else:
        print("ERROR: Check solver paths and input file formats.")


def get_joint_wmc_node_name(assignment_dict, mapping, m_name):
    assignment_vars = []
    for node, state in assignment_dict.items():
        if node in mapping and state in mapping[node]:
            assignment_vars.append(mapping[node][state])
        else:
            print(f"Error: {node}={state} not found in mapping.")
            return 0.0

    temp_file = "temp_res/temp_map_query.wcnf"
    if create_wcnf_file(temp_file, assignment_vars, m_name):
        weight = run_wmc(temp_file)
        return weight
    return 0.0


if __name__ == "__main__":
    MODEL = "asia"
    with open(f"temp_res/{MODEL}_data.json", "r") as f:
        data = json.load(f)
        mapping = data["mapping"]

    # query_probability_node_int([3], 15, MODEL)
    query_probability_node_name({"bronc": "yes", "lung": "no"}, 'dysp', 'yes', mapping, MODEL)

    # MODEL = "net_parents_5"
    # with open(f"temp_res/{MODEL}_data.json", "r") as f:
    #     data = json.load(f)
    #     mapping = data["mapping"]

    # # query_probability_node_int([3], 15, MODEL)
    # query_probability_node_name({}, 'v2', 's1', mapping, MODEL)