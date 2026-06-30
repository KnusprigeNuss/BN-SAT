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
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from bn_to_mpe_sat import generate_mpe_encoding


warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)
sys.setrecursionlimit(20000)


TIMEOUT_SECONDS = 300  
MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024  
UWRMAXSAT_BIN = os.path.expanduser("~/uwrmaxsat/build/release/bin/uwrmaxsat")
NUM_PER_TIER = 5
MODELS = ["hailfinder", "win95pts", "andes", "link", "munin2", "munin", "pathfinder"]
MAX_HILL_CLIMB_ITERS = 10000


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


# uwrmaxsat
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


# ve
def run_ve_mpe_safe(ve_infer, evidence, model):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    joint_prob = None
    try:
        mpe_nodes = list(set(ve_infer.variables) - set(evidence.keys()))
        mpe_state = ve_infer.map_query(variables=mpe_nodes, evidence=evidence, show_progress=False)
        
        full_state = evidence.copy()
        full_state.update(mpe_state)
        joint_prob = get_fast_joint_log_prob(model, full_state)
        
        return time.perf_counter() - start, "SUCCESS", joint_prob
    except PythonTimeout:
        return float('inf'), "T/O", None
    except MemoryError:
        return float('inf'), "MEM_OUT", None
    except Exception:
        return float('inf'), "ERR", None
    finally:
        signal.alarm(0)


# calc for log prob
def get_fast_joint_log_prob(model, state):
    log_prob = 0.0
    for node in model.nodes():
        cpd = model.get_cpds(node)
        evidence_vals = [state[parent] for parent in cpd.variables[1:]]
        val = cpd.get_value(**{node: state[node]}, **dict(zip(cpd.variables[1:], evidence_vals)))
        
        if val == 0: return float('-inf')
        log_prob += np.log10(val)
    return log_prob


# hill climb alg
def run_hill_climbing_safe(model, evidence, max_iters=1000):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    try:
        mpe_nodes = [n for n in model.nodes() if n not in evidence]
        if not mpe_nodes:
            return time.perf_counter() - start, "SUCCESS", get_fast_joint_log_prob(model, evidence)
            
        global_best_prob = float('-inf')
        
        for _ in range(50):
            current_state = {}
            current_prob = float('-inf')
            
            for _ in range(20):
                temp_state = evidence.copy()
                for node in mpe_nodes:
                    temp_state[node] = random.choice(model.get_cpds(node).state_names[node])
                
                score = get_fast_joint_log_prob(model, temp_state)
                if score > float('-inf'):
                    current_state = temp_state
                    current_prob = score
                    break
                    
            if current_prob == float('-inf'):
                continue

            # steepest asc algo
            for _ in range(max_iters):
                best_neighbor = current_state.copy()
                best_prob = current_prob
                improved = False

                # scan neighbors and flip one node
                for node_to_flip in mpe_nodes:
                    states = model.get_cpds(node_to_flip).state_names[node_to_flip]
                    for state in states:
                        if state == current_state[node_to_flip]: continue
                            
                        neighbor_state = current_state.copy()
                        neighbor_state[node_to_flip] = state
                        neighbor_prob = get_fast_joint_log_prob(model, neighbor_state)
                        
                        if neighbor_prob > best_prob:
                            best_prob = neighbor_prob
                            best_neighbor = neighbor_state
                            improved = True

                if not improved: 
                    break 
                    
                current_state = best_neighbor
                current_prob = best_prob
            
            if current_prob > global_best_prob:
                global_best_prob = current_prob
                
        if global_best_prob == float('-inf'):
            return float('inf'), "INV", float('-inf')
            
        return time.perf_counter() - start, "SUCCESS", global_best_prob
    except PythonTimeout:
        return float('inf'), "T/O", None
    except MemoryError:
        return float('inf'), "MEM_OUT", None
    except Exception:
        return float('inf'), "ERR", None
    finally:
        signal.alarm(0)


def get_tier_display(times, statuses):
    valid_times = [t for t, s in zip(times, statuses) if s in ["SUCCESS", "MM"]]
    
    mem_outs = statuses.count("MEM_OUT")
    timeouts = statuses.count("T/O")
    invalids = statuses.count("INV")
    mismatches = statuses.count("MM")
    errs = statuses.count("ERR")
    
    if not valid_times:
        if mem_outs > 0: return "MEM_OUT"
        if timeouts > 0: return "T/O"
        if invalids > 0: return "INV"
        return "ERR"
        
    avg = sum(valid_times) / len(valid_times)
    
    extras = []
    if mem_outs > 0: extras.append(f"{mem_outs} MO")
    if timeouts > 0: extras.append(f"{timeouts} TO")
    if invalids > 0: extras.append(f"{invalids} INV")
    if mismatches > 0: extras.append(f"{mismatches} MM")
    if errs > 0: extras.append(f"{errs} ERR")
    
    if extras:
        return f"{avg:.4f} ({', '.join(extras)})"
    return f"{avg:.4f}"


def mpe_final_tournament():
    # maybe with seed?
    random.seed(42) 
    np.random.seed(42) 
    set_global_memory_limit()
    mismatches = []
    
    print("\n" + "=" * 95)
    print(f"{'Model':<15} | {'Tier':<10} | {'UWrMaxSat (s)':<15} | {'Var. Elim (s)':<15} | {'Hill Climb (s)':<15}")
    print("-" * 95)

    for m_name in MODELS:
        model = get_example_model(m_name)
        ve_infer = VariableElimination(model)
        tiered_queries = generate_dynamic_mpe_queries(m_name, NUM_PER_TIER)
        
        for tier_name, test_queries in tiered_queries.items():
            # print("starting tier: " + str(tier_name))
            uwr_times, ve_times, hc_times = [], [], []
            uwr_stats, ve_stats, hc_stats = [], [], []
            
            for evidence in test_queries:
                out_prefix = f"temp_res/{m_name}_mpe"
                o_pref = f"{m_name}"
                model = get_example_model(m_name)
                mpe_nodes = [n for n in model.nodes() if n not in evidence]
                
                # uwrmaxsat
                generate_mpe_encoding(model, evidence, o_pref)
                wcnf_path = f"{out_prefix}.wcnf"
                t, stat, uwr_cost = run_uwrmaxsat_safe(wcnf_path)
                uwr_times.append(t); uwr_stats.append(stat)
                
                # ve
                t, stat, ve_prob = run_ve_mpe_safe(ve_infer, evidence, model)
                ve_times.append(t); ve_stats.append(stat)
                
                # hc
                t, stat, hc_prob = run_hill_climbing_safe(model, evidence, MAX_HILL_CLIMB_ITERS)

                if stat == "SUCCESS":
                    if hc_prob == float('-inf'):
                        stat = "INV"
                        t = float('inf')
                    elif uwr_cost is not None:
                        SCALE = 1000000
                        expected_hc_cost = int(round(-hc_prob * SCALE))
                        # print(expected_hc_cost)
                        # print(uwr_cost)
                        if abs(expected_hc_cost - uwr_cost) > 20:
                            stat = "MM"

                # hc_times.append(t); hc_stats.append(stat)
                # if hc_prob == float('-inf'):
                #     hc_times.append(float('inf'))
                #     hc_stats.append("INV")
                # else:
                hc_times.append(t)
                hc_stats.append(stat)

                if uwr_cost is not None and ve_prob is not None and ve_prob != float('-inf'):
                    SCALE = 1000000
                    expected_uwr_cost = int(round(-ve_prob * SCALE))
                    
                    if abs(expected_uwr_cost - uwr_cost) > 20:
                        err_msg = f"{m_name} ({tier_name}) | UWrMaxSat: {uwr_cost} vs VE Expected: {expected_uwr_cost}"
                        print(f"\nMISMATCH DETECTED {err_msg}")
                        mismatches.append(err_msg)
                
                # print("uwr    :" + str(uwr_cost))
                # print("exp uwr:" + str(expected_uwr_cost))
                # print("ve_prob:" + str(ve_prob))
                # print("hc_prob:" + str(hc_prob))

            uwr_display = get_tier_display(uwr_times, uwr_stats)
            ve_display = get_tier_display(ve_times, ve_stats)
            hc_display = get_tier_display(hc_times, hc_stats)
            
            print(f"{m_name:<15} | {tier_name:<10} | {uwr_display:<15} | {ve_display:<15} | {hc_display:<15}")

    print("=" * 95)
    if mismatches:
        print(f"Found {len(mismatches)} deviations.")
    else:
        print("Verification complete. No mismatches detected.")

if __name__ == "__main__":
    mpe_final_tournament()