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
import itertools
import re
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
from bn_to_map import generate_map_cnf


warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)
sys.setrecursionlimit(20000)


TIMEOUT_SECONDS = 300
MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024 
GANAK_BIN = "./solvers_bin/ganak"
NUM_PER_TIER = 5
MODELS = ["asia", "cancer", "sachs", "alarm", "child", "water", "barley", "hailfinder", "win95pts", "andes", "link", "pathfinder", "munin2", "munin"]


def set_global_memory_limit():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except ValueError:
        pass


class PythonTimeout(Exception): pass
def timeout_handler(signum, frame): raise PythonTimeout()


def generate_dynamic_map_queries(model_name, num_per_tier=5):
    model = get_example_model(model_name)
    nodes = list(model.nodes())
    total_nodes = len(nodes)
    
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_per_tier, show_progress=False)
    
    queries = {
        "fewM_fewE": [], 
        "fewM_muchE": [], 
        "muchM_fewE": [], 
        "muchM_muchE": []
    }
    
    few_ev_count = max(1, int(total_nodes * 0.15))
    much_ev_count = max(2, int(total_nodes * 0.40))
    few_map_count = max(2, min(3, int(total_nodes * 0.05)))
    much_map_count = max(4, min(7, int(total_nodes * 0.15))) 
    
    for i in range(num_per_tier):
        sample = samples.iloc[i].to_dict()
        shuffled_nodes = nodes.copy()
        random.shuffle(shuffled_nodes)
        
        few_map_nodes = shuffled_nodes[:few_map_count]
        rem_nodes_fewM = shuffled_nodes[few_map_count:]
        queries["fewM_fewE"].append((few_map_nodes, {n: sample[n] for n in rem_nodes_fewM[:few_ev_count]}))
        queries["fewM_muchE"].append((few_map_nodes, {n: sample[n] for n in rem_nodes_fewM[:much_ev_count]}))

        much_map_nodes = shuffled_nodes[:much_map_count]
        rem_nodes_muchM = shuffled_nodes[much_map_count:]
        queries["muchM_fewE"].append((much_map_nodes, {n: sample[n] for n in rem_nodes_muchM[:few_ev_count]}))
        queries["muchM_muchE"].append((much_map_nodes, {n: sample[n] for n in rem_nodes_muchM[:much_ev_count]}))
    return queries


# ganak multi query map
def run_ganak_multiquery_safe(m_name, map_nodes, evidence, data, model):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    best_log_prob = float('-inf')
    
    try:
        mapping = data["mapping"]
        weights = data["weights"]
        clauses = data["clauses"]
        num_vars = data["num_vars"]
        
        target_states = [model.get_cpds(n).state_names[n] for n in map_nodes]
        combinations = list(itertools.product(*target_states))
        
        temp_file = f"temp_res/{m_name}_ganak_query.cnf"
        
        if len(combinations) > 500: return float('inf'), "T/O", None
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
            res = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=TIMEOUT_SECONDS,
                preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
            )
            
            raw_prob = 0.0
            for line in res.stdout.split('\n'):
                if "exact quadruple float" in line:
                    raw_prob = float(line.split()[-1])
                    break
            
            if raw_prob > 0:
                log_prob = np.log10(raw_prob)
                if log_prob > best_log_prob:
                    best_log_prob = log_prob

        return time.perf_counter() - start, "SUCCESS", best_log_prob
    except PythonTimeout:
        return float('inf'), "T/O", None
    except Exception as e:
        if "MemoryError" in str(type(e)): return float('inf'), "MEM_OUT", None
        return float('inf'), "ERR", None
    finally:
        signal.alarm(0)


# dpmc map
def run_dpmc_safe(cnf_path, mapping, map_nodes):
    start = time.perf_counter()
    log_prob = None
    try:
        # sed for float zeros
        subprocess.run(f"sed -i 's/0\\.000000000000000/0.00000000000000000000001/g' {cnf_path}", shell=True)
        subprocess.run(f"sed -i 's/ 0\\.0 / 0.00000000000000000000001 /g' {cnf_path}", shell=True)
        
        cmd = (
            f'./solvers_bin/DPMC/bin/lg "./solvers_bin/DPMC/bin/flow_cutter_pace17 -p 100" < {cnf_path} | '
            f'./solvers_bin/DPMC/bin/dmc --cf={cnf_path} --wc=1 --pc=1 --dp=c --er=1 --mf=2'
        )
        
        res = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=TIMEOUT_SECONDS,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        )
        
        prob_match = re.search(r"exact double prec-sci\s+([eE\d\.-]+)", res.stdout)
        if prob_match: 
            raw_prob = float(prob_match.group(1))
            log_prob = np.log10(raw_prob) if raw_prob > 0 else float('-inf')
                
        return time.perf_counter() - start, "SUCCESS", log_prob
    except subprocess.TimeoutExpired:
        return float('inf'), "T/O", None
    except Exception as e:
        if "MemoryError" in str(type(e)): return float('inf'), "MEM_OUT", None
        return float('inf'), "ERR", None


# ve map
def run_ve_map_safe(ve_infer, map_nodes, evidence):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    log_prob = None
    try:
        best_state = ve_infer.map_query(variables=map_nodes, evidence=evidence, show_progress=False)
        
        p_map_given_e = 1.0
        if map_nodes:
            cond_factor = ve_infer.query(variables=map_nodes, evidence=evidence, show_progress=False)
            cond_factor.reduce([(k, best_state[k]) for k in map_nodes])
            p_map_given_e = float(cond_factor.values.item() if hasattr(cond_factor.values, 'item') else cond_factor.values)
            
        p_e = 1.0
        if evidence:
            ev_vars = list(evidence.keys())
            ev_factor = ve_infer.query(variables=ev_vars, show_progress=False)
            ev_factor.reduce([(k, evidence[k]) for k in ev_vars])
            p_e = float(ev_factor.values.item() if hasattr(ev_factor.values, 'item') else ev_factor.values)
            
        raw_prob = p_map_given_e * p_e
        log_prob = np.log10(raw_prob) if raw_prob > 0 else float('-inf')

        return time.perf_counter() - start, "SUCCESS", log_prob
    except PythonTimeout:
        return float('inf'), "T/O", None
    except MemoryError:
        return float('inf'), "MEM_OUT", None
    except Exception as e:
        return float('inf'), "ERR", None
    finally:
        signal.alarm(0)


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
    
    if mem_outs > 0:
        return f"{avg:.4f} ({mem_outs} MO)"
    if timeouts > 0:
        return f"{avg:.4f} ({timeouts} TO)"
    return f"{avg:.4f}"


def map_final_tournament():
    # maybe with seed?
    random.seed(42) 
    np.random.seed(42) 
    set_global_memory_limit()
    mismatches = []
    
    print("\n" + "=" * 95)
    print(f"{'Model':<15} | {'Tier':<15} | {'Ganak-Spam (s)':<15} | {'DPMC (s)':<15} | {'Var. Elim (s)':<15}")
    print("-" * 95)

    for m_name in MODELS:
        model = get_example_model(m_name)
        ve_infer = VariableElimination(model)
        tiered_queries = generate_dynamic_map_queries(m_name, NUM_PER_TIER)
        
        for tier_name, test_queries in tiered_queries.items():
            ganak_times, dpmc_times, ve_times = [], [], []
            ganak_stats, dpmc_stats, ve_stats = [], [], []
            
            for map_nodes, evidence in test_queries:
                out_prefix = f"temp_res/{m_name}_map"
                cnf_path = f"{out_prefix}.cnf"
                data_path = f"{out_prefix}_data.json"
                
                generate_map_cnf(m_name, map_nodes, evidence, out_prefix)
                with open(data_path, "r") as f:
                    data = json.load(f)
                    mapping = data["mapping"]
                
                # multi query ganak
                t, stat, ganak_prob = run_ganak_multiquery_safe(m_name, map_nodes, evidence, data, model)
                ganak_times.append(t); ganak_stats.append(stat)

                # dpmc 
                t, stat, dpmc_prob = run_dpmc_safe(cnf_path, mapping, map_nodes)
                dpmc_times.append(t); dpmc_stats.append(stat)
                
                # ve
                t, stat, ve_prob = run_ve_map_safe(ve_infer, map_nodes, evidence)
                ve_times.append(t); ve_stats.append(stat)
                
                # veri
                if ve_prob is not None and ve_prob != float('-inf'):
                    if ganak_prob is not None and ganak_prob != float('-inf') and abs(ganak_prob - ve_prob) > 0.0001:
                        err_msg = f"{m_name} ({tier_name}) | Ganak: {ganak_prob:.5f} vs VE: {ve_prob:.5f}"
                        print(f"\nMISMATCH DETECTED {err_msg}")
                        mismatches.append(err_msg)
                    
                    if dpmc_prob is not None and dpmc_prob != float('-inf') and abs(dpmc_prob - ve_prob) > 0.0001:
                        err_msg = f"{m_name} ({tier_name}) | DPMC: {dpmc_prob:.5f} vs VE: {ve_prob:.5f}"
                        print(f"\nMISMATCH DETECTED {err_msg}")
                        mismatches.append(err_msg)
                    # print("ve_prob: " + str(ve_prob))
                    # print("ganak_prob: " + str(ganak_prob))
                    # print("dpmc_prob: " + str(dpmc_prob))

            ganak_display = get_tier_display(ganak_times, ganak_stats)
            dpmc_display = get_tier_display(dpmc_times, dpmc_stats)
            ve_display = get_tier_display(ve_times, ve_stats)
            
            print(f"{m_name:<15} | {tier_name:<15} | {ganak_display:<15} | {dpmc_display:<15} | {ve_display:<15}")
            # print(f"{m_name:<15} | {tier_name:<15} | {0:<15} | {dpmc_display:<15} | {0:<15}")

    print("=" * 95)
    if mismatches:
        print(f"Found {len(mismatches)} probability deviations.")
    else:
        print("Verification complete. No mismatches detected.")

if __name__ == "__main__":
    map_final_tournament()