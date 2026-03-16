import subprocess
import os
import time
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination
import itertools

MODEL_NAME = "alarm"
SOLVER_PATH = os.path.expanduser("~/sharpsat-td-main/bin/sharpSAT")
BASE_DIR = "/mnt/c/Users/alexn/Desktop/BN-SAT/temp_res"
BASE_CNF = os.path.join(BASE_DIR, f"{MODEL_NAME}.cnf")
BASE_WMC = os.path.join(BASE_DIR, f"{MODEL_NAME}.wmc")

model = get_example_model(MODEL_NAME)
inference = VariableElimination(model)

mapping = {}
var_count = 1
for node in model.nodes():
    mapping[node] = {}
    states = model.get_cpds(node).state_names[node]
    for state in states:
        mapping[node][state] = var_count
        var_count += 1

def get_pgmpy_result(node, state, evidence):
    res = inference.query(variables=[node], evidence=evidence, show_progress=False)
    state_idx = model.get_cpds(node).get_state_no(node, state)
    return float(res.values[state_idx])

def run_sat_query(evidence_dict, query_node, query_state):
    evidence_lits = [mapping[n][s] for n, s in evidence_dict.items()]
    query_lit = mapping[query_node][query_state]
    
    create_wcnf_file("temp_res/verify_denom.wcnf", evidence_lits)
    denom = run_wmc("temp_res/verify_denom.wcnf")
    
    create_wcnf_file("temp_res/verify_num.wcnf", evidence_lits + [query_lit])
    num = run_wmc("temp_res/verify_num.wcnf")
    
    return num / denom if denom and denom > 0 else 0.0

def create_wcnf_file(output_name, extra_clauses=[]):
    with open(BASE_CNF, 'r') as f:
        lines = f.readlines()
    num_vars = int(lines[0].split()[2])
    cnf_body = [l.strip() for l in lines if not l.startswith(('p', 'c')) and l.strip()]
    
    with open(BASE_WMC, 'r') as f:
        weights = {int(l.split()[1]): l.split()[2] for l in f if l.startswith('w')}

    with open(output_name, 'w') as f:
        f.write(f"p cnf {num_vars} {len(cnf_body) + len(extra_clauses)}\n")
        for line in cnf_body: f.write(f"{line}\n")
        for lit in extra_clauses: f.write(f"{lit} 0\n")
        for i in range(1, num_vars + 1):
            f.write(f"c p weight {i} {weights.get(i, '1.0')} 0\n")
            f.write(f"c p weight -{i} {weights.get(-i, '1.0')} 0\n")

def run_wmc(file_name):
    abs_path = os.path.abspath(file_name)
    cmd = ["./sharpSAT", "-WE", "-decot", "5", "-tmpdir", ".", "-prec", "15", abs_path]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(SOLVER_PATH))
    for line in res.stdout.split('\n'):
        if "exact arb float" in line: return float(line.split()[-1])
    return 0.0






queries = [
    {"node": "INTUBATION", "state": "NORMAL", "ev": {"SHUNT": "NORMAL", "PRESS": "ZERO"}},
    {"node": "VENTLUNG", "state": "ZERO", "ev": {"VENTMACH": "LOW", "VENTTUBE": "LOW", "EXPCO2": "ZERO"}},
    {"node": "PVSAT", "state": "LOW", "ev": {"VENTALV": "LOW", "SAO2": "HIGH"}},
    {"node": "HYPOVOLEMIA", "state": "TRUE", "ev": {"LVEDVOLUME": "NORMAL", "STROKEVOLUME": "LOW"}}
]

print(f"{'Query Node':<15} | {'State':<15} | {'Evidence':<70} | {'pgmpy (VE)':<12} | {'SAT (WMC)':<12} | {'Match'}")
print("-" * 124)

for q in queries:
    ve_val = get_pgmpy_result(q['node'], q['state'], q['ev'])
    sat_val = run_sat_query(q['ev'], q['node'], q['state'])
    ev_str = str(q['ev']) if q['ev'] else "None"
    match = "yes" if abs(ve_val - sat_val) < 1e-6 else "no"
    
    print(f"{q['node']:<15} | {q['state']:<15} | {ev_str:<70} | {ve_val:<12.4f} | {sat_val:<12.4f} | {match}")