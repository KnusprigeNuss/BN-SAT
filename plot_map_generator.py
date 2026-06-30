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
import numpy as np
import itertools

warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)
sys.setrecursionlimit(20000)

from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from bn_to_map import generate_map_cnf

MODEL_NAME = "alarm" 
NUM_QUERIES = 100
TIMEOUT_SECONDS = 60
MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024  
CSV_FILENAME = "map_benchmark_data_100.csv"
GANAK_BIN = "./solvers_bin/ganak"


class PythonTimeout(Exception): pass
def timeout_handler(signum, frame): raise PythonTimeout()


def run_ganak_multiquery_safe(m_name, map_nodes, evidence, data, model):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    
    try:
        mapping = data["mapping"]
        weights = data["weights"]
        clauses = data["clauses"]
        num_vars = data["num_vars"]
        
        target_states = [model.get_cpds(n).state_names[n] for n in map_nodes]
        combinations = list(itertools.product(*target_states))
        
        temp_file = f"temp_res/{m_name}_ganak_query.cnf"
        
        if len(combinations) > 500: return "T/O" 
        
        for combo in combinations:
            current_evidence_lits = []
            
            for node, state in zip(map_nodes, combo):
                current_evidence_lits.append(mapping[node][state])
            
            for node, state in evidence.items():
                current_evidence_lits.append(mapping[node][state])
                
            with open(temp_file, "w") as f:
                f.write(f"p cnf {num_vars} {len(clauses) + len(current_evidence_lits)}\n")
                
                for c in clauses:
                    f.write(" ".join(map(str, c)) + " 0\n")
                for c in current_evidence_lits:
                    f.write(f"{c} 0\n")
                    
                for var, prob in weights.items():
                    f.write(f"c p weight {var} {float(prob):.15f} 0\n")
                    f.write(f"c p weight -{var} 1.000000 0\n")

            cmd = [GANAK_BIN, "--mode", "1", temp_file]
            subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=TIMEOUT_SECONDS,
                preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
            )

        return time.perf_counter() - start
    except PythonTimeout:
        return "T/O"
    except Exception as e:
        if "MemoryError" in str(type(e)): return "MEM_OUT"
        return "ERR"
    finally:
        signal.alarm(0)


def run_dpmc_safe(cnf_path):
    start = time.perf_counter()
    try:
        subprocess.run(f"sed -i 's/0\\.000000000000000/0.00000000000000000000001/g' {cnf_path}", shell=True)
        cmd = (f'./solvers_bin/DPMC/bin/lg "./solvers_bin/DPMC/bin/flow_cutter_pace17 -p 100" < {cnf_path} | '
               f'./solvers_bin/DPMC/bin/dmc --cf={cnf_path} --wc=1 --pc=1 --dp=c --er=1 --mf=2')
        subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                       preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES)))
        return time.perf_counter() - start
    except subprocess.TimeoutExpired: return "T/O"
    except Exception as e:
        if "MemoryError" in str(type(e)): return "MEM_OUT"
        return "ERR"


def run_ve_map_safe(ve_infer, map_nodes, evidence):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    try:
        ve_infer.map_query(variables=map_nodes, evidence=evidence, show_progress=False)
        return time.perf_counter() - start
    except PythonTimeout: return "T/O"
    except MemoryError: return "MEM_OUT"
    except Exception: return "ERR"
    finally: signal.alarm(0)


def run_benchmarks():
    print(f"--- Generating {NUM_QUERIES} MAP queries for {MODEL_NAME} ---")
    model = get_example_model(MODEL_NAME)
    ve_infer = VariableElimination(model)
    nodes = list(model.nodes())
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=NUM_QUERIES, show_progress=False)
    
    results = []
    
    for i in range(NUM_QUERIES):
        sample = samples.iloc[i].to_dict()
        shuffled_nodes = nodes.copy()
        random.shuffle(shuffled_nodes)
        
        map_count = random.randint(2, 5)
        ev_count = random.randint(2, 10)
        
        map_nodes = shuffled_nodes[:map_count]
        evidence = {n: sample[n] for n in shuffled_nodes[map_count:map_count+ev_count]}
        
        print(f"\n[{i+1}/{NUM_QUERIES}] MAP Query | {len(map_nodes)} MAP | {len(evidence)} EV")
        row_data = {'Query_ID': i+1, 'Map_Size': len(map_nodes), 'Ev_Size': len(evidence)}
        
        out_prefix = f"temp_res/{MODEL_NAME}_map"
        generate_map_cnf(MODEL_NAME, map_nodes, evidence, out_prefix)
        cnf_path = f"{out_prefix}.cnf"
        data_path = f"{out_prefix}_data.json"
        

        with open(data_path, "r") as f:
            data = json.load(f)
        
        row_data['Ganak'] = run_ganak_multiquery_safe(MODEL_NAME, map_nodes, evidence, data, model)
        print(f"  -> Ganak (Spam): {row_data['Ganak']}")
        
        row_data['DPMC'] = run_dpmc_safe(cnf_path)
        print(f"  -> DPMC: {row_data['DPMC']}")
        
        row_data['Var_Elim'] = run_ve_map_safe(ve_infer, map_nodes, evidence)
        print(f"  -> Var. Elim: {row_data['Var_Elim']}")

        results.append(row_data)
        
        with open(CSV_FILENAME, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Query_ID', 'Map_Size', 'Ev_Size', 'Ganak', 'DPMC', 'Var_Elim'])
            writer.writeheader()
            writer.writerows(results)


def generate_plots():
    if not os.path.exists(CSV_FILENAME): return
    data = list(csv.DictReader(open(CSV_FILENAME, 'r')))
    
    # cactus
    plt.figure(figsize=(9, 6))
    solvers = {'Ganak': '#ff7f0e', 'DPMC': '#d62728', 'Var_Elim': '#9467bd'}
    
    for solver, color in solvers.items():
        times = [float(row[solver]) for row in data if row[solver] not in ['T/O', 'ERR', 'MEM_OUT'] and float(row[solver]) > 0]
        times.sort()
        plt.plot(range(1, len(times) + 1), times, marker='o', markersize=4, linewidth=2, label=solver.replace('_', ' '), color=color)

    plt.yscale('log')
    plt.xlabel('Number of Solved Instances', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (seconds) [Log Scale]', fontsize=12, fontweight='bold')
    plt.title(f'Cactus Plot: MAP Inference on {MODEL_NAME.capitalize()} Network', fontsize=14, pad=15)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig('map_cactus_plot.png', dpi=300)

    # scatter
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    def extract_coords(s1, s2):
        x, y = [], []
        for r in data:
            if r[s1] in ['ERR', 'MEM_OUT'] or r[s2] in ['ERR', 'MEM_OUT']: continue
            x.append(float(r[s1]) if r[s1] != 'T/O' else 60.0)
            y.append(float(r[s2]) if r[s2] != 'T/O' else 60.0)
        return x, y

    # dpmc vs ganak
    x, y = extract_coords('DPMC', 'Ganak')
    axs[0].scatter(x, y, alpha=0.7, color='red', edgecolors='k')
    axs[0].set_title('Architectural Impact: Proj. Model Counting vs Multi-Query Spam', pad=10)
    axs[0].set_xlabel('DPMC Time (s)', fontweight='bold')
    axs[0].set_ylabel('Ganak (Spam) Time (s)', fontweight='bold')

    # dpmc vs ve
    x, y = extract_coords('DPMC', 'Var_Elim')
    axs[1].scatter(x, y, alpha=0.7, color='purple', edgecolors='k')
    axs[1].set_title('Projected Compilation vs Variable Elimination', pad=10)
    axs[1].set_xlabel('DPMC Time (s)', fontweight='bold')
    axs[1].set_ylabel('Variable Elimination Time (s)', fontweight='bold')
    
    # ganak vs ve
    x, y = extract_coords('Ganak', 'Var_Elim')
    axs[2].scatter(x, y, alpha=0.7, color='orange', edgecolors='k')
    axs[2].set_title('Multi-Query vs Variable Elimination', pad=10)
    axs[2].set_xlabel('Ganak (Spam) Time (s)', fontweight='bold')
    axs[2].set_ylabel('Variable Elimination Time (s)', fontweight='bold')

    for ax in axs:
        ax.plot([0.001, 100], [0.001, 100], 'k--', label='Tie (y=x)')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.005, 70); ax.set_ylim(0.005, 70)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('map_scatter_plots.png', dpi=300)

if __name__ == "__main__":
    if not os.path.exists(CSV_FILENAME): run_benchmarks()
    generate_plots()