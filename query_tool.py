import subprocess
import os

MODEL = "alarm"
SOLVER_PATH = os.path.expanduser("~/sharpsat-td-main/bin/sharpSAT")
BASE_DIR = "/mnt/c/Users/alexn/Desktop/BN-SAT/temp_res"
BASE_CNF = os.path.join(BASE_DIR, MODEL + ".cnf")
BASE_WMC = os.path.join(BASE_DIR, MODEL + ".wmc")

def create_wcnf_file(output_name, extra_clauses=[]):
    with open(BASE_CNF, 'r') as f:
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
    
    with open(BASE_WMC, 'r') as f:
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
    solver_dir = os.path.dirname(SOLVER_PATH)
    
    # from sharpsat repo
    cmd = ["./sharpSAT", "-WE", "-decot", "5", "-tmpdir", ".", "-prec", "10", abs_file_path]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=solver_dir)

    for line in result.stdout.split('\n'):
        if "exact arb float" in line:
            return float(line.split()[-1])
    return None



def query_probability(evidence_list, query_node_var):
    print(f"Starting query for P({query_node_var} | {evidence_list})...")
    
    create_wcnf_file("temp_res/temp_denom.wcnf", evidence_list)
    denom = run_wmc("temp_res/temp_denom.wcnf")
    # print(f"Denominator result: {denom}")
    
    create_wcnf_file("temp_res/temp_num.wcnf", evidence_list + [query_node_var])
    num = run_wmc("temp_res/temp_num.wcnf")
    # print(f"Numerator result: {num}")
    
    if denom is not None and num is not None:
        if denom == 0: return "P(Evidence) is 0 - logically impossible."
        prob = (num / denom) * 100
        print(f"SUCCESS: P({query_node_var} | {evidence_list}) = {prob:.2f}%\n")
        return prob
    else:
        print("ERROR: Check solver paths and input file formats.")



query_probability([], 2)
# query_probability([], 77)
# query_probability([], 78)
# query_probability([], 79)