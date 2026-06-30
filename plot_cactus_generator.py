import os
import sys
import time
import json
import random
import signal
import resource
import csv
import matplotlib.pyplot as plt
import warnings
import logging
import subprocess

sys.setrecursionlimit(20000)
warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)

from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination

from bn_to_cnf_wmc import generate_wmc_cnf as build_baseline
from bn_to_cnf_wmc_w_pruning import generate_wmc_cnf as build_pruned
from query_sharpsat import query_probability_node_name
from query_ganak import query_probability_ganak
from query_nnf import solve_circuit


MODEL_NAME = "alarm"
NUM_QUERIES = 100
TIMEOUT_SECONDS = 60
MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024 
CSV_FILENAME = "benchmark_data_100.csv"


def limit_memory():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except ValueError:
        pass


class PythonTimeout(Exception): pass
def timeout_handler(signum, frame): raise PythonTimeout()


def run_with_timeout(func, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    try:
        func(*args, **kwargs)
        t = time.perf_counter() - start
        return t
    except PythonTimeout:
        return "T/O"
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return "ERR"
    finally:
        signal.alarm(0)


def evaluate_d4_query(nnf_lines, ev_dict, q_node, q_state, base_mapping, cpt_weights):
    num = solve_circuit(nnf_lines, {**ev_dict, q_node: q_state}, base_mapping, cpt_weights)
    den = solve_circuit(nnf_lines, ev_dict, base_mapping, cpt_weights)
    if den == 0: return 0.0
    return (num / den) * 100


def run_ve_query(ve_infer, q_node, ev_dict):
    factor = ve_infer.query(variables=[q_node], evidence=ev_dict, show_progress=False)
    return True


def generate_gauntlet_queries(model_name, n=100):
    model = get_example_model(model_name)
    nodes = list(model.nodes())
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=n, show_progress=False)
    
    queries = []
    for i in range(n):
        sample = samples.iloc[i].to_dict()
        shuffled_nodes = nodes.copy()
        random.shuffle(shuffled_nodes)
        
        num_evidence = random.randint(1, 20)
        query_node = shuffled_nodes[0]
        query_state = sample[query_node]
        
        evidence_dict = {n: sample[n] for n in shuffled_nodes[1:num_evidence+1]}
        queries.append((query_node, query_state, evidence_dict))
    return queries


def run_benchmarks():
    print(f"--- Generating {NUM_QUERIES} queries for {MODEL_NAME} ---")
    queries = generate_gauntlet_queries(MODEL_NAME, NUM_QUERIES)
    model = get_example_model(MODEL_NAME)
    ve_infer = VariableElimination(model)
    
    results = []
    
    print("Compiling d4 circuit")
    build_baseline(model, MODEL_NAME, debug=False)
    d4_bin = os.path.expanduser("~/d4/d4")
    nnf_path = f"temp_res/{MODEL_NAME}.nnf"
    
    if os.path.exists(nnf_path):
        os.remove(nnf_path)
        
    d4_cmd = f"{d4_bin} temp_res/{MODEL_NAME}.cnf -dDNNF -out={nnf_path}"
    print(f"Executing: {d4_cmd}")
    os.system(d4_cmd)
    
    if not os.path.exists(nnf_path) or os.path.getsize(nnf_path) == 0:
        print(f"Error: d4 still failed to compile.")
        return

    with open(nnf_path, "r") as f:
        raw_nnf_lines = f.readlines()
    nnf_lines = [line for line in raw_nnf_lines if not line.strip().startswith(('nnf', 'c'))]
    
    with open(f"temp_res/{MODEL_NAME}_data.json", "r") as f:
        data = json.load(f)
        cpt_weights = {int(k): float(v) for k, v in data["weights"].items()}
    
    for i, (q_node, q_state, ev_dict) in enumerate(queries):
        print(f"\n[{i+1}/{NUM_QUERIES}] Query: P({q_node} | {len(ev_dict)} evidence)")
        row_data = {'Query_ID': i+1, 'Ev_Size': len(ev_dict)}
        

        t_ve = run_with_timeout(run_ve_query, ve_infer, q_node, ev_dict)
        row_data['Var_Elim'] = t_ve
        print(f"  -> Var. Elim: {t_ve}")


        build_baseline(model, MODEL_NAME, debug=False)
        with open(f"temp_res/{MODEL_NAME}_data.json", "r") as f:
            base_mapping = json.load(f)["mapping"]
            
        # ss
        t_ss = run_with_timeout(query_probability_node_name, ev_dict, q_node, q_state, base_mapping, MODEL_NAME)
        row_data['SharpSAT'] = t_ss
        print(f"  -> SharpSAT:  {t_ss}")

        # ganak
        t_gn = run_with_timeout(query_probability_ganak, ev_dict, q_node, q_state, base_mapping, MODEL_NAME)
        row_data['Ganak'] = t_gn
        print(f"  -> Ganak:     {t_gn}")
        
        # d4
        t_d4 = run_with_timeout(evaluate_d4_query, nnf_lines, ev_dict, q_node, q_state, base_mapping, cpt_weights)
        row_data['d4'] = t_d4
        print(f"  -> d4 Eval:   {t_d4}")
        
        # pruned ss
        build_pruned(MODEL_NAME, [q_node], ev_dict, out_prefix=f"temp_res/{MODEL_NAME}")
        with open(f"temp_res/{MODEL_NAME}_data.json", "r") as f:
            pruned_mapping = json.load(f)["mapping"]
        t_ss_p = run_with_timeout(query_probability_node_name, ev_dict, q_node, q_state, pruned_mapping, MODEL_NAME)
        row_data['Pruned_SharpSAT'] = t_ss_p
        print(f"  -> Pruned SS: {t_ss_p}")

        results.append(row_data)
        
        with open(CSV_FILENAME, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Query_ID', 'Ev_Size', 'SharpSAT', 'Ganak', 'd4', 'Pruned_SharpSAT', 'Var_Elim'])
            writer.writeheader()
            writer.writerows(results)


def generate_plots():
    if not os.path.exists(CSV_FILENAME):
        print("No CSV data found. Run benchmarks first.")
        return
        
    data = []
    with open(CSV_FILENAME, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: data.append(row)

    print("\n--- Generating Plots ---")
    
    # cactus
    plt.figure(figsize=(9, 6))
    solvers = {
        'SharpSAT': '#1f77b4', 
        'Ganak': '#ff7f0e', 
        'd4': '#d62728', 
        'Pruned_SharpSAT': '#2ca02c',
        'Var_Elim': '#9467bd' 
    }
    
    for solver, color in solvers.items():
        times = [float(row[solver]) for row in data if row[solver] not in ['T/O', 'ERR'] and float(row[solver]) > 0]
        times.sort()
        
        label_name = solver.replace('_', ' ')
        if solver == 'Var_Elim':
            label_name = "Classical (Var. Elim)"
            
        plt.plot(range(1, len(times) + 1), times, marker='o', markersize=4, linewidth=2, label=label_name, color=color)

    plt.yscale('log')
    plt.xlabel('Number of Solved Instances', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (seconds) [Log Scale]', fontsize=12, fontweight='bold')
    plt.title('Cactus Plot: Marginal Inference on Alarm Network', fontsize=14, pad=15)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig('cactus_plot.png', dpi=300)
    print("Saved 'cactus_plot.png'")

    # scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ganak vs ss
    x_gn, y_ss = [], []
    for row in data:
        if row['Ganak'] == 'ERR' or row['SharpSAT'] == 'ERR': continue
        t_g = float(row['Ganak']) if row['Ganak'] != 'T/O' else 60.0
        t_s = float(row['SharpSAT']) if row['SharpSAT'] != 'T/O' else 60.0
        x_gn.append(t_g); y_ss.append(t_s)

    ax1.scatter(x_gn, y_ss, alpha=0.7, color='purple', edgecolors='k')
    ax1.plot([0.001, 100], [0.001, 100], 'k--', label='Tie (y=x)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlim(0.005, 70); ax1.set_ylim(0.005, 70)
    ax1.set_xlabel('Ganak Time (s)', fontweight='bold')
    ax1.set_ylabel('SharpSAT Time (s)', fontweight='bold')
    ax1.set_title('Architectural Comparison: SharpSAT vs Ganak\n(Dots below diagonal = SharpSAT is faster)', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # pruned ss vs base ss
    x_pruned, y_base = [], []
    for row in data:
        if row['Pruned_SharpSAT'] == 'ERR' or row['SharpSAT'] == 'ERR': continue
        t_p = float(row['Pruned_SharpSAT']) if row['Pruned_SharpSAT'] != 'T/O' else 60.0
        t_b = float(row['SharpSAT']) if row['SharpSAT'] != 'T/O' else 60.0
        x_pruned.append(t_p); y_base.append(t_b)

    ax2.scatter(x_pruned, y_base, alpha=0.7, color='teal', edgecolors='k')
    ax2.plot([0.001, 100], [0.001, 100], 'k--', label='Tie (y=x)')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlim(0.005, 70); ax2.set_ylim(0.005, 70)
    ax2.set_xlabel('Pruned SharpSAT Time (s)', fontweight='bold')
    ax2.set_ylabel('Baseline SharpSAT Time (s)', fontweight='bold')
    ax2.set_title('Structural Impact: Baseline vs Barren Pruning\n(Dots ABOVE diagonal = Pruned is faster!)', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('scatter_plots.png', dpi=300)
    print("Saved 'scatter_plots.png'")

if __name__ == "__main__":
    if not os.path.exists(CSV_FILENAME):
        run_benchmarks()
    
    generate_plots()