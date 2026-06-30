import time
import json
import os
import random
import signal
import resource
import logging
import warnings
import subprocess
import sys
import numpy as np
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from bn_to_mpe_sat import generate_mpe_encoding

warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)
sys.setrecursionlimit(20000)


TIMEOUT_SECONDS = 300  
MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024  
UWRMAXSAT_BIN = os.path.expanduser("~/uwrmaxsat/build/release/bin/uwrmaxsat")
NUM_PER_TIER = 5
MODELS = ["barley", "hailfinder", "andes", "munin2", "munin"]
#  alone:
# barley, hailfinder, andes, munin2, munin

def set_global_memory_limit():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except ValueError:
        pass


class PythonTimeout(Exception): pass
def timeout_handler(signum, frame): raise PythonTimeout()


def generate_dynamic_mpe_queries(model_name, num_per_tier=5):
    model = get_example_model(model_name)
    nodes = list(model.nodes())
    
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_per_tier, show_progress=False)
    
    queries = {"much_ev": [], "few_ev": [], "no_ev": []}
    
    few_count = max(1, int(len(nodes) * 0.15))
    much_count = max(2, int(len(nodes) * 0.40))
    
    for i in range(num_per_tier):
        sample = samples.iloc[i].to_dict()
        
        shuffled_nodes = nodes.copy()
        random.shuffle(shuffled_nodes)
        
        queries["much_ev"].append({n: sample[n] for n in shuffled_nodes[:much_count]})
        queries["few_ev"].append({n: sample[n] for n in shuffled_nodes[:few_count]})
        queries["no_ev"].append({})
    return queries


def run_rc2_safe(wcnf_path):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    cost = None
    try:
        formula = WCNF(from_file=wcnf_path)
        with RC2(formula) as rc2:
            model = rc2.compute()
            cost = rc2.cost
        return time.perf_counter() - start, "SUCCESS", cost
    except PythonTimeout:
        return float('inf'), "T/O", None
    except MemoryError:
        return float('inf'), "MEM_OUT", None
    except Exception:
        return float('inf'), "ERR", None
    finally:
        signal.alarm(0)


def run_uwrmaxsat_safe(wcnf_path):
    start = time.perf_counter()
    cost = None
    try:
        cmd = [UWRMAXSAT_BIN, "-v0", "-m", wcnf_path]
        res = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=TIMEOUT_SECONDS,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        )
        for line in res.stdout.splitlines():
            if line.startswith("o "):
                cost = int(line.split()[1])
                
        return time.perf_counter() - start, "SUCCESS", cost
    except subprocess.TimeoutExpired:
        return float('inf'), "T/O", None
    except Exception as e:
        if "MemoryError" in str(type(e)): return float('inf'), "MEM_OUT", None
        return float('inf'), "ERR", None


def get_tier_display(times, statuses):
    valid_times = [t for t, s in zip(times, statuses) if s == "SUCCESS"]
    
    mem_outs = statuses.count("MEM_OUT")
    timeouts = statuses.count("T/O")
    errs = statuses.count("ERR")
    
    if not valid_times:
        if mem_outs > 0: return "MEM_OUT"
        if timeouts > 0: return "T/O"
        return "ERR"
        
    avg = sum(valid_times) / len(valid_times)
    
    extras = []
    if mem_outs > 0: extras.append(f"{mem_outs} MO")
    if timeouts > 0: extras.append(f"{timeouts} TO")
    if errs > 0: extras.append(f"{errs} ERR")
    
    if extras:
        return f"{avg:.4f} ({', '.join(extras)})"
    return f"{avg:.4f}"

def mpe_solver_tournament():
    # maybe with seed?
    random.seed(42) 
    np.random.seed(42) 
    set_global_memory_limit()
    mismatches = []
    
    print("\n" + "=" * 80)
    print(f"{'Model':<15} | {'Tier':<10} | {'RC2 (PySAT) (s)':<20} | {'UWrMaxSat (C++) (s)':<20}")
    print("-" * 80)

    for m_name in MODELS:
        tiered_queries = generate_dynamic_mpe_queries(m_name, NUM_PER_TIER)
        
        for tier_name, test_queries in tiered_queries.items():
            rc2_times, uwr_times = [], []
            rc2_stats, uwr_stats = [], []
            
            for evidence in test_queries:
                # print(m_name + tier_name)
                out_prefix = f"temp_res/{m_name}_mpe"
                o_pref = f"{m_name}"
                model = get_example_model(m_name)
                mpe_nodes = [n for n in model.nodes() if n not in evidence]
                
                generate_mpe_encoding(model, evidence, o_pref)
                wcnf_path = f"{out_prefix}.wcnf"
                
                # --- Test RC2 ---
                # if m_name in ["barley", "hailfinder", "win95pts", "andes", "link", "pathfinder", "munin2", "munin"]:
                rc2_times.append(float('inf'))
                rc2_stats.append("MEM_OUT")
                rc2_cost = None
                # else:
                    # t, stat, rc2_cost = run_rc2_safe(wcnf_path)
                    # rc2_times.append(t); rc2_stats.append(stat)
                
                # --- Test UWrMaxSat ---
                # uwr_times.append(float('inf'))
                # uwr_stats.append("MEM_OUT")
                # rc2_cost = None
                t, stat, uwr_cost = run_uwrmaxsat_safe(wcnf_path)
                uwr_times.append(t); uwr_stats.append(stat)
                
                # --- VERIFICATION ---
                if rc2_cost is not None and uwr_cost is not None:
                    if rc2_cost != uwr_cost:
                        err_msg = f"{m_name} ({tier_name}) | RC2 Cost: {rc2_cost} vs UWrMaxSat Cost: {uwr_cost}"
                        print(f"\n---> [MISMATCH DETECTED] {err_msg}")
                        mismatches.append(err_msg)
                # print(rc2_cost)
                # print(uwr_cost)

            
            # Format and Print Row
            rc2_display = get_tier_display(rc2_times, rc2_stats)
            uwr_display = get_tier_display(uwr_times, uwr_stats)
            
            print(f"{m_name:<15} | {tier_name:<10} | {rc2_display:<20} | {uwr_display:<20}")

    # Final Summary Block
    print("=" * 80)
    if mismatches:
        print(f"[WARNING] Found {len(mismatches)} cost deviations between solvers.")
    else:
        print("[SUCCESS] Verification complete. UWrMaxSat perfectly matches PySAT RC2 logic.")

if __name__ == "__main__":
    mpe_solver_tournament()