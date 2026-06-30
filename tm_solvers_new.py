import time
import subprocess
import os
import json
import random
import resource
import warnings
import logging
import signal
import numpy as np
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)

class PythonTimeout(Exception): 
    pass

def timeout_handler(signum, frame):
    raise PythonTimeout()

from query_nnf import solve_circuit
from query_sharpsat import query_probability_node_name
from query_ganak import query_probability_ganak
from bn_to_cnf_wmc import generate_wmc_cnf

TIMEOUT_SECONDS = 300 
MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024 

def limit_memory():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except ValueError:
        pass


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


def run_sharpsat_safe(evidence, query_node, query_state, mapping, m_name):
    start = time.perf_counter()
    try:
        res = query_probability_node_name(evidence, query_node, query_state, mapping, m_name)
        return res, time.perf_counter() - start, "SUCCESS"
    except subprocess.TimeoutExpired:
        return 0.0, time.perf_counter() - start, "T/O"
    except Exception as e:
        if "MemoryError" in str(type(e)):
            return 0.0, time.perf_counter() - start, "MEM_OUT"
        return 0.0, time.perf_counter() - start, "ERR"


def run_ganak_safe(evidence, query_node, query_state, mapping, m_name):
    start = time.perf_counter()
    try:
        res = query_probability_ganak(evidence, query_node, query_state, mapping, m_name)
        return res, time.perf_counter() - start, "SUCCESS"
    except subprocess.TimeoutExpired:
        return 0.0, time.perf_counter() - start, "T/O"
    except Exception as e:
        if "MemoryError" in str(type(e)):
            return 0.0, time.perf_counter() - start, "MEM_OUT"
        return 0.0, time.perf_counter() - start, "ERR"


def generate_dynamic_queries(model_name, num_per_tier=5):
    model = get_example_model(model_name)
    nodes = list(model.nodes())
    total_nodes = len(nodes)
    
    with open(f"temp_res/{model_name}_data.json", "r") as f:
        data = json.load(f)
        mapping = data["mapping"]
    
    few_count = max(1, int(total_nodes * 0.15))
    much_count = max(2, int(total_nodes * 0.40))
    
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_per_tier, show_progress=False)
    
    queries = {"no_ev": [], "few_ev": [], "much_ev": []}
    
    for i in range(num_per_tier):
        sample = samples.iloc[i].to_dict()
        
        valid_sample = {}
        for k, v in sample.items():
            if k in mapping and v in mapping[k]:
                valid_sample[k] = v
                
        if not valid_sample:
            continue 
            
        valid_nodes = list(valid_sample.keys())
        
        query_node = random.choice(valid_nodes)
        query_state = valid_sample[query_node]
        
        remaining_nodes = [n for n in valid_nodes if n != query_node]
        random.shuffle(remaining_nodes)
        
        queries["no_ev"].append(({}, query_node, query_state))
        
        few_nodes = remaining_nodes[:min(few_count, len(remaining_nodes))]
        ev_few = {n: valid_sample[n] for n in few_nodes}
        queries["few_ev"].append((ev_few, query_node, query_state))
        
        much_nodes = remaining_nodes[:min(much_count, len(remaining_nodes))]
        ev_much = {n: valid_sample[n] for n in much_nodes}
        queries["much_ev"].append((ev_much, query_node, query_state))
    return queries


def run_command(cmd, input_str=None):
    start = time.perf_counter()
    proc = subprocess.run(cmd, input=input_str, capture_output=True, text=True, shell=False)
    end = time.perf_counter()
    return proc.stdout, end - start


def get_ground_truth(model_name, query_node, evidence, query_state):
    model = get_example_model(model_name)
    infer = VariableElimination(model)
    res_factor = infer.query(variables=[query_node], evidence=evidence, show_progress=False)
    res_factor.normalize() 
    state_idx = model.get_cpds(query_node).get_state_no(query_node, query_state)
    return res_factor.values[state_idx]


def tournament(model_list):
    random.seed(42) 
    np.random.seed(42) 
    
    print("\n" + "=" * 120)
    print(f"{'Model':<15} | {'Tier':<10} | {'sharpSAT (s)':<15} | {'Ganak (s)':<15} | {'d4 Eval (s)':<15} | {'d4 Comp (s)':<15}")
    print("-" * 120)

    for m_name in model_list:
        out_prefix = f"temp_res/{m_name}"
        
        model = get_example_model(m_name)
        generate_wmc_cnf(model, m_name, debug=False)
        with open(f"{out_prefix}_data.json", "r") as f:
            data = json.load(f)
            mapping = data["mapping"]
            cpt_weights = {int(k): v for k, v in data["weights"].items()}
        
        # d4-comp
        start_comp = time.perf_counter()
        d4_comp_stat = "SUCCESS"
        d4_comp_time = 0.0
        
        try:
            subprocess.run(
                [SOLVERS["d4"], f"{out_prefix}.cnf", "-dDNNF", f"-out={out_prefix}.nnf"], 
                stdout=subprocess.DEVNULL,
                timeout=TIMEOUT_SECONDS,
                preexec_fn=limit_memory
            )
            d4_comp_time = time.perf_counter() - start_comp
        except subprocess.TimeoutExpired:
            d4_comp_stat = "T/O"
        except Exception as e:
            if "MemoryError" in str(type(e)):
                d4_comp_stat = "MEM_OUT"
            else:
                d4_comp_stat = "ERR"

        tiered_queries = generate_dynamic_queries(m_name, num_per_tier=5)

        for tier_name, test_queries in tiered_queries.items():
            # if m_name == "munin2" and tier_name == "no_ev":
            #     print("skipping munin2 no ev")
            #     continue
            
            ss_times, gn_times, d4_eval_times = [], [], []
            ss_stats, gn_stats, d4_eval_stats = [], [], []
            
            for evidence, query_node, query_state in test_queries:
                # print("query")
                
                # ss
                ss_res, ss_t, ss_stat = run_sharpsat_safe(evidence, query_node, query_state, mapping, m_name)
                ss_times.append(ss_t); ss_stats.append(ss_stat)
                
                # ganak
                gn_res, gn_t, gn_stat = run_ganak_safe(evidence, query_node, query_state, mapping, m_name)
                gn_times.append(gn_t); gn_stats.append(gn_stat)
                
                # d4
                if d4_comp_stat == "SUCCESS":
                    start_eval = time.perf_counter()
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(TIMEOUT_SECONDS)
                    try:
                        with open(f"temp_res/{m_name}.nnf", "r") as f: 
                            lines = f.readlines()
                        p_num = solve_circuit(lines, {**evidence, query_node: query_state}, mapping, cpt_weights)
                        p_den = solve_circuit(lines, evidence, mapping, cpt_weights)
                        d4_res = p_num / p_den if p_den != 0 else 0.0
                        d4_eval_times.append(time.perf_counter() - start_eval)
                        d4_eval_stats.append("SUCCESS")
                    except PythonTimeout:
                        d4_eval_times.append(0.0)
                        d4_eval_stats.append("T/O")
                    except Exception as e:
                        if "MemoryError" in str(type(e)):
                            d4_eval_times.append(0.0)
                            d4_eval_stats.append("MEM_OUT")
                        else:
                            d4_eval_times.append(0.0)
                            d4_eval_stats.append("ERR")
                    finally:
                        signal.alarm(0)
                else:
                    d4_eval_times.append(0.0)
                    d4_eval_stats.append(d4_comp_stat)

            ss_str = get_tier_display(ss_times, ss_stats)
            gn_str = get_tier_display(gn_times, gn_stats)
            d4_str = get_tier_display(d4_eval_times, d4_eval_stats)
            d4_comp_str = f"{d4_comp_time:.4f}" if d4_comp_stat == "SUCCESS" else d4_comp_stat

            print(f"{m_name:<15} | {tier_name:<10} | {ss_str:<15} | {gn_str:<15} | {d4_str:<15} | {d4_comp_str:<15}")


if __name__ == "__main__":
    SOLVERS = {
        "sharpsat": os.path.expanduser("~/sharpsat-td-main/bin/sharpSAT"),
        "ganak": "./solvers_bin/ganak",
        "d4": os.path.expanduser("~/d4/d4")
    }
    
    MODELS = ["barley"]
    
    tournament(MODELS)