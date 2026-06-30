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
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)
sys.setrecursionlimit(20000)

from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from bn_to_mpe_sat import generate_mpe_encoding

MODEL_NAME = "alarm" 
NUM_QUERIES = 100
TIMEOUT_SECONDS = 60
MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024  
CSV_FILENAME = "mpe_benchmark_data_100.csv"
UWRMAXSAT_BIN = os.path.expanduser("~/uwrmaxsat/build/release/bin/uwrmaxsat")


class PythonTimeout(Exception): pass
def timeout_handler(signum, frame): raise PythonTimeout()


def get_fast_joint_log_prob(model, state):
    log_prob = 0.0
    for node in model.nodes():
        cpd = model.get_cpds(node)
        evidence_vals = [state[parent] for parent in cpd.variables[1:]]
        val = cpd.get_value(**{node: state[node]}, **dict(zip(cpd.variables[1:], evidence_vals)))
        if val == 0: return float('-inf')
        log_prob += np.log10(val)
    return log_prob


def run_uwrmaxsat_safe(wcnf_path):
    start = time.perf_counter()
    try:
        cmd = [UWRMAXSAT_BIN, "-v0", "-m", wcnf_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                             preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES)))
        return time.perf_counter() - start
    except subprocess.TimeoutExpired:
        return "T/O"
    except Exception as e:
        if "MemoryError" in str(type(e)): return "MEM_OUT"
        return "ERR"


def run_rc2_safe(wcnf_path):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    try:
        formula = WCNF(from_file=wcnf_path)
        with RC2(formula) as rc2:
            rc2.compute()
        return time.perf_counter() - start
    except PythonTimeout: return "T/O"
    except MemoryError: return "MEM_OUT"
    except Exception: return "ERR"
    finally: signal.alarm(0)


def run_ve_mpe_safe(ve_infer, evidence, model):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    try:
        mpe_nodes = list(set(ve_infer.variables) - set(evidence.keys()))
        ve_infer.map_query(variables=mpe_nodes, evidence=evidence, show_progress=False)
        return time.perf_counter() - start
    except PythonTimeout: return "T/O"
    except MemoryError: return "MEM_OUT"
    except Exception: return "ERR"
    finally: signal.alarm(0)


def run_hill_climbing_safe(model, evidence):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    try:
        mpe_nodes = [n for n in model.nodes() if n not in evidence]
        if not mpe_nodes: return time.perf_counter() - start
        
        current_state = evidence.copy()
        for node in mpe_nodes:
            current_state[node] = random.choice(model.get_cpds(node).state_names[node])
            
        for _ in range(100): 
            improved = False
            best_prob = get_fast_joint_log_prob(model, current_state)
            best_neighbor = current_state.copy()
            
            for node_to_flip in mpe_nodes:
                for state in model.get_cpds(node_to_flip).state_names[node_to_flip]:
                    if state == current_state[node_to_flip]: continue
                    neighbor = current_state.copy()
                    neighbor[node_to_flip] = state
                    prob = get_fast_joint_log_prob(model, neighbor)
                    if prob > best_prob:
                        best_prob = prob
                        best_neighbor = neighbor
                        improved = True
            if not improved: break
            current_state = best_neighbor
            
        if best_prob == float('-inf'): return "INV"
        return time.perf_counter() - start
    except PythonTimeout: return "T/O"
    except Exception: return "ERR"
    finally: signal.alarm(0)


def run_benchmarks():
    print(f"--- Generating {NUM_QUERIES} MPE queries for {MODEL_NAME} ---")
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
        
        num_evidence = random.randint(1, int(len(nodes) * 0.40))
        evidence = {n: sample[n] for n in shuffled_nodes[:num_evidence]}
        
        print(f"\n[{i+1}/{NUM_QUERIES}] MPE Query | {len(evidence)} evidence nodes")
        row_data = {'Query_ID': i+1, 'Ev_Size': len(evidence)}
        
        out_prefix = f"temp_res/{MODEL_NAME}_mpe"
        generate_mpe_encoding(model, evidence, MODEL_NAME)
        wcnf_path = f"{out_prefix}.wcnf"
        
        row_data['UWrMaxSat'] = run_uwrmaxsat_safe(wcnf_path)
        print(f"  -> UWrMaxSat: {row_data['UWrMaxSat']}")
        
        row_data['RC2'] = run_rc2_safe(wcnf_path)
        print(f"  -> RC2 (PySAT): {row_data['RC2']}")
        
        row_data['Var_Elim'] = run_ve_mpe_safe(ve_infer, evidence, model)
        print(f"  -> Var. Elim: {row_data['Var_Elim']}")
        
        row_data['Hill_Climb'] = run_hill_climbing_safe(model, evidence)
        print(f"  -> Hill Climb: {row_data['Hill_Climb']}")

        results.append(row_data)
        
        with open(CSV_FILENAME, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Query_ID', 'Ev_Size', 'UWrMaxSat', 'RC2', 'Var_Elim', 'Hill_Climb'])
            writer.writeheader()
            writer.writerows(results)


def generate_plots():
    if not os.path.exists(CSV_FILENAME): return
    data = list(csv.DictReader(open(CSV_FILENAME, 'r')))
    
    # cactus
    plt.figure(figsize=(9, 6))
    solvers = {'UWrMaxSat': '#1f77b4', 'RC2': '#2ca02c', 'Var_Elim': '#9467bd', 'Hill_Climb': '#ff7f0e'}
    
    for solver, color in solvers.items():
        times = [float(row[solver]) for row in data if row[solver] not in ['T/O', 'ERR', 'MEM_OUT', 'INV'] and float(row[solver]) > 0]
        times.sort()
        plt.plot(range(1, len(times) + 1), times, marker='o', markersize=4, linewidth=2, label=solver.replace('_', ' '), color=color)

    plt.yscale('log')
    plt.xlabel('Number of Solved Instances', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (seconds) [Log Scale]', fontsize=12, fontweight='bold')
    plt.title(f'Cactus Plot: MPE Inference on {MODEL_NAME.capitalize()} Network', fontsize=14, pad=15)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig('mpe_cactus_plot.png', dpi=300)

    # scatter plots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    def extract_coords(s1, s2):
        x, y = [], []
        for r in data:
            if r[s1] in ['ERR', 'MEM_OUT', 'INV'] or r[s2] in ['ERR', 'MEM_OUT', 'INV']: continue
            x.append(float(r[s1]) if r[s1] != 'T/O' else 60.0)
            y.append(float(r[s2]) if r[s2] != 'T/O' else 60.0)
        return x, y

    # uwrmaxsat vs rc2
    x, y = extract_coords('UWrMaxSat', 'RC2')
    axs[0].scatter(x, y, alpha=0.7, color='teal', edgecolors='k')
    axs[0].set_title('Architectural Impact: Native C++ vs PySAT FFI', pad=10)
    axs[0].set_xlabel('UWrMaxSat Time (s)', fontweight='bold')
    axs[0].set_ylabel('RC2 (PySAT) Time (s)', fontweight='bold')

    # uwrmaxsat vs ve
    x, y = extract_coords('UWrMaxSat', 'Var_Elim')
    axs[1].scatter(x, y, alpha=0.7, color='purple', edgecolors='k')
    axs[1].set_title('Paradigm Shift: MaxSAT vs Variable Elimination', pad=10)
    axs[1].set_xlabel('UWrMaxSat Time (s)', fontweight='bold')
    axs[1].set_ylabel('Variable Elimination Time (s)', fontweight='bold')
    
    # uwrmaxsat vs hc
    x, y = extract_coords('UWrMaxSat', 'Hill_Climb')
    axs[2].scatter(x, y, alpha=0.7, color='orange', edgecolors='k')
    axs[2].set_title('Exact vs Approximate: MaxSAT vs Hill Climbing', pad=10)
    axs[2].set_xlabel('UWrMaxSat Time (s)', fontweight='bold')
    axs[2].set_ylabel('Hill Climbing Time (s)', fontweight='bold')

    for ax in axs:
        ax.plot([0.001, 100], [0.001, 100], 'k--', label='Tie (y=x)')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.005, 70); ax.set_ylim(0.005, 70)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('mpe_scatter_plots.png', dpi=300)

if __name__ == "__main__":
    if not os.path.exists(CSV_FILENAME): run_benchmarks()
    generate_plots()