import time
import json
import os
import random
import signal
import resource
import logging
import warnings
import subprocess
import numpy as np
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from bn_to_cnf_wmc_w_pruning import generate_wmc_cnf
from query_sharpsat import get_joint_wmc_node_name

warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)


TIMEOUT_SECONDS = 300
MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024  
NUM_PER_TIER = 5
MODELS = ["link"]


def set_global_memory_limit():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except ValueError:
        pass


class PythonTimeout(Exception): pass
def timeout_handler(signum, frame): raise PythonTimeout()


def run_python_with_timeout(func, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    start = time.perf_counter()
    res = None
    status = "SUCCESS"
    try:
        res = func(*args, **kwargs)
    except PythonTimeout:
        status = "T/O"
    except MemoryError:
        status = "MEM_OUT"
    except Exception:
        status = "ERR"
    finally:
        signal.alarm(0) 
        elapsed = time.perf_counter() - start
        
    return res, elapsed, status


def generate_dynamic_queries_with_maxtree(model_name, num_per_tier=3):
    model = get_example_model(model_name)
    nodes = list(model.nodes())
    leaves = [n for n in model.nodes() if model.out_degree(n) == 0]
    roots = [n for n in model.nodes() if model.in_degree(n) == 0]
    
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_per_tier, show_progress=False)
    
    queries = {"no_ev": [], "few_ev": [], "much_ev": [], "max_tree": []}
    
    few_count = max(1, int(len(nodes) * 0.15))
    much_count = max(2, int(len(nodes) * 0.40))
    
    for i in range(num_per_tier):
        sample = samples.iloc[i].to_dict()
        
        query_node = random.choice(nodes)
        query_state = sample[query_node]
        rem_nodes = [n for n in nodes if n != query_node]
        random.shuffle(rem_nodes)
        
        queries["no_ev"].append(({}, query_node, query_state))
        queries["few_ev"].append(({n: sample[n] for n in rem_nodes[:few_count]}, query_node, query_state))
        queries["much_ev"].append(({n: sample[n] for n in rem_nodes[:much_count]}, query_node, query_state))
        
        root_query = random.choice(roots) if roots else query_node
        root_state = sample[root_query]
        leaf_ev = {l: sample[l] for l in leaves if l != root_query}
        queries["max_tree"].append((leaf_ev, root_query, root_state))
        
    return queries


def solve_baseline_pgmpy_wrapped(inference, target_state, evidence):
    query_vars = list(target_state.keys())
    
    if evidence:
        cond_factor = inference.query(variables=query_vars, evidence=evidence, show_progress=False)
        cond_factor.reduce([(k, target_state[k]) for k in query_vars])
        p_q_given_e = float(cond_factor.values.item() if hasattr(cond_factor.values, 'item') else cond_factor.values)
        
        ev_vars = list(evidence.keys())
        ev_factor = inference.query(variables=ev_vars, show_progress=False)
        ev_factor.reduce([(k, evidence[k]) for k in ev_vars])
        p_e = float(ev_factor.values.item() if hasattr(ev_factor.values, 'item') else ev_factor.values)
        
        return p_q_given_e * p_e
    else:
        factor = inference.query(variables=query_vars, show_progress=False)
        factor.reduce([(k, target_state[k]) for k in query_vars])
        return float(factor.values.item() if hasattr(factor.values, 'item') else factor.values)


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


def run_pruning_tournament():
    random.seed(42) 
    np.random.seed(42)
    set_global_memory_limit()
    
    mismatches = []

    print("\n" + "=" * 90)
    print(f"{'Model':<15} | {'Tier':<10} | {'sharpSAT (s)':<20} | {'Var. Elimination (s)':<20}")
    print("-" * 90)
    
    for m_name in MODELS:
        model = get_example_model(m_name)
        ve_infer = VariableElimination(model)

        tiered_queries = generate_dynamic_queries_with_maxtree(m_name, num_per_tier=NUM_PER_TIER)

        for tier_name, test_queries in tiered_queries.items():
            ss_times, ve_times = [], []
            ss_stats, ve_stats = [], []
            
            for evidence, query_node, query_state in test_queries:
                target_state = {query_node: query_state}
                out_prefix = f"temp_res/{m_name}" 
                
                # generate pruned cnf
                _, _, gen_status = run_python_with_timeout(
                    generate_wmc_cnf, 
                    model_name=m_name, 
                    query_nodes=[query_node], 
                    evidence=evidence, 
                    out_prefix=out_prefix
                )
                
                if gen_status != "SUCCESS":
                    ss_times.append(0.0); ss_stats.append(gen_status)
                    prob_ss = None
                else:
                    data_path = f"{out_prefix}_data.json"
                    try:
                        with open(data_path, "r") as f:
                            mapping = json.load(f)["mapping"]
                        
                        # ss
                        ss_res, ss_t, ss_stat = run_python_with_timeout(get_joint_wmc_node_name, target_state, mapping, m_name)
                        ss_times.append(ss_t); ss_stats.append(ss_stat)
                        prob_ss = ss_res if ss_stat == "SUCCESS" else None
                    except Exception:
                        ss_times.append(0.0); ss_stats.append("ERR")
                        prob_ss = None

                # ve
                ve_res, ve_t, ve_stat = run_python_with_timeout(solve_baseline_pgmpy_wrapped, ve_infer, target_state, evidence)
                ve_times.append(ve_t); ve_stats.append(ve_stat)
                prob_ve = ve_res if ve_stat == "SUCCESS" else None

                
                print("VE: " + str(prob_ve) + ", SHARPSAT: " + str(prob_ss))
                if prob_ss is not None and prob_ve is not None:
                    if round(prob_ss, 5) != round(prob_ve, 5):
                        err_msg = f"{m_name} ({tier_name}) | sharpSAT: {prob_ss:.6f} vs VE: {prob_ve:.6f}"
                        print(f"\n---> [MISMATCH DETECTED] {err_msg}")
                        mismatches.append(err_msg)

            ss_display = get_tier_display(ss_times, ss_stats)
            ve_display = get_tier_display(ve_times, ve_stats)
            
            print(f"{m_name:<15} | {tier_name:<10} | {ss_display:<20} | {ve_display:<20}")

    print("=" * 90)
    if mismatches:
        print(f"Found {len(mismatches)} mathematical deviations during pruning tests.")
    else:
        print("Verification complete. Pruned CNF logic perfectly matches exact VE calculations.")

if __name__ == "__main__":
    run_pruning_tournament()