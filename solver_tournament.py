import time
import subprocess
import os
import json
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination
from query_nnf import solve_circuit
from query_sharpsat import query_probability_node_name
from query_ganak import query_probability_ganak


SOLVERS = {
    "sharpsat": os.path.expanduser("~/sharpsat-td-main/bin/sharpSAT"),
    "ganak": "./solvers_bin/ganak",
    "d4": os.path.expanduser("~/d4/d4")
}
MODELS = ["asia"] # , "alarm", "hailfinder"
QUERIES_PER_MODEL = 3



def run_command(cmd, input_str=None):
    start = time.perf_counter()
    proc = subprocess.run(cmd, input=input_str, capture_output=True, text=True, shell=False)
    end = time.perf_counter()
    return proc.stdout, end - start

def benchmark_sharpsat(cnf_path):
    cmd = [SOLVERS["sharpsat"], "-WE", "-prec", "10", cnf_path]
    stdout, duration = run_command(cmd)
    # Parse: 'exact arb float'
    return duration

def benchmark_ganak(cnf_path):
    cmd = [SOLVERS["ganak"], "--mode", "1", cnf_path]
    stdout, duration = run_command(cmd)
    return duration

def benchmark_d4_compile(cnf_path, nnf_path):
    cmd = [SOLVERS["d4"], cnf_path, "-dDNNF", f"-out={nnf_path}"]
    stdout, duration = run_command(cmd)
    return duration

def run_ganak_inference(model_name, evidence, query_node, query_state, mapping):
    start = time.perf_counter()
    # p_num = run_ganak_raw(model_name, {**evidence, query_node: query_state})
    # p_den = run_ganak_raw(model_name, evidence)
    duration = time.perf_counter() - start
    return duration

def get_ground_truth(model_name, query_node, evidence, query_state):
    model = get_example_model(model_name)
    infer = VariableElimination(model)
    res = infer.query(variables=[query_node], evidence=evidence)
    state_idx = model.get_cpds(query_node).get_state_no(query_node, query_state)
    return res.values[state_idx]


    

results = {}
def tournament(model_list):
    for m_name in model_list:
        print(f"\n🚀 TOURNAMENT: {m_name}")
        results[m_name] = {"sharpsat": [], "ganak": [], "d4_compile": 0, "d4_eval": []}
        with open(f"temp_res/{m_name}_data.json", "r") as f:
            data = json.load(f)
            mapping = data["mapping"]
            cpt_weights = {int(k): v for k, v in data["weights"].items()}

        start_comp = time.perf_counter()
        subprocess.run([SOLVERS["d4"], f"temp_res/{m_name}.cnf", "-dDNNF", f"-out=temp_res/{m_name}.nnf"], stdout=subprocess.DEVNULL)
        comp_time = time.perf_counter() - start_comp
        results[m_name]["d4_compile"] = time.perf_counter() - start_comp

        # query_node = 'dysp'
        # query_state = 'no'
        # evidence = {'tub': 'yes'}

        test_queries = [
            ({'tub': 'yes'}, 'dysp', 'yes'),
            ({'smoke': 'yes'}, 'lung', 'yes'),
            ({'asia': 'yes'}, 'tub', 'yes')
        ]

        for evidence, query_node, query_state in test_queries:
            ground_truth = get_ground_truth(m_name, query_node, evidence, query_state)
            
            with open(f"temp_res/{m_name}.nnf", "r") as f: lines = f.readlines()
            start_eval = time.perf_counter()
            p_num = solve_circuit(lines, {**evidence, query_node: query_state}, mapping, cpt_weights)
            p_den = solve_circuit(lines, evidence, mapping, cpt_weights)
            d4_res = p_num / p_den
            results[m_name]["d4_eval"].append(time.perf_counter() - start_eval)

            start_ss = time.perf_counter()
            query_probability_node_name(evidence, query_node, query_state, mapping, m_name)
            results[m_name]["sharpsat"].append(time.perf_counter() - start_ss)

            start_ganak = time.perf_counter()
            query_probability_ganak(evidence, query_node, query_state, mapping, m_name)
            results[m_name]["ganak"].append(time.perf_counter() - start_ganak)

            
        
    print("\n" + "="*75)
    print(f"{'Model':<12} | {'Solver':<12} | {'Avg Query (s)':<15} | {'Setup/Comp (s)'}")
    print("-" * 75)
    for m in MODELS:
        ss_avg = sum(results[m]["sharpsat"]) / QUERIES_PER_MODEL
        gn_avg = sum(results[m]["ganak"]) / QUERIES_PER_MODEL
        d4_avg = sum(results[m]["d4_eval"]) / QUERIES_PER_MODEL
        
        print(f"{m:<12} | sharpSAT-td | {ss_avg:<15.4f} | {'N/A'}")
        print(f"{'':<12} | Ganak       | {gn_avg:<15.4f} | {'N/A'}")
        print(f"{'':<12} | d4 (KC)     | {d4_avg:<15.6f} | {results[m]['d4_compile']:.4f}")
        print("-" * 75)


tournament(["asia"])