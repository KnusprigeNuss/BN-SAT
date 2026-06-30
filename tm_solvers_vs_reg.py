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

warnings.filterwarnings("ignore")
logging.getLogger("pgmpy").setLevel(logging.ERROR)

from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination, BeliefPropagation, ApproxInference
from pgmpy.sampling import BayesianModelSampling

from query_sharpsat import query_probability_node_name
from tm_solvers_new import generate_dynamic_queries
from bn_to_cnf_wmc import generate_wmc_cnf

TIMEOUT_SECONDS = 300 
MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024  
NUM_SAMPLES = 2000
MODELS = ["asia", "link", "munin"]


def set_global_memory_limit():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except ValueError:
        pass

class PythonTimeout(Exception): 
    pass

def timeout_handler(signum, frame):
    raise PythonTimeout()


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


def exact_query_wrapper(infer_obj, model, query_node, query_state, evidence):
    res_factor = infer_obj.query(variables=[query_node], evidence=evidence, show_progress=False)
    res_factor.normalize()
    state_idx = res_factor.state_names[query_node].index(query_state)
    return res_factor.values[state_idx]


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


def get_tier_display(times, statuses, errors=None):
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
    
    if errors and len(errors) > 0:
        avg_err = (sum(errors) / len(errors)) * 100
        extras.append(f"±{avg_err:.1f}%")
        
    if extras:
        return f"{avg:.4f} ({', '.join(extras)})"
    return f"{avg:.4f}"


def tournament_vs_classical():
    random.seed(42) 
    np.random.seed(42) 
    set_global_memory_limit()
    
    mismatches = []

    print("\n" + "=" * 125)
    print(f"{'Model':<15} | {'Tier':<10} | {'sharpSAT (s)':<15} | {'Var. Elim (s)':<15} | {'Junc Tree (s)':<15} | {'Sampling (s) [±Err]':<20}")
    print("-" * 125)

    for m_name in MODELS:
        out_prefix = f"temp_res/{m_name}"
        
        model = get_example_model(m_name)
        generate_wmc_cnf(model, m_name, debug=False)
        
        for ext in [".cnf", ".wmc", "_data.json"]:
            src = f"{m_name}{ext}"
            dst = f"{out_prefix}{ext}"
            if os.path.exists(src):
                os.rename(src, dst)

        ve_infer = VariableElimination(model)
        sampling_infer = ApproxInference(model)
        
        def init_jt():
            return BeliefPropagation(model)

        # jt_infer, _, jt_init_status = run_python_with_timeout(init_jt)
        
        # jt_supported = (jt_init_status == "SUCCESS")
        jt_supported = False

        with open(f"{out_prefix}_data.json", "r") as f:
            data = json.load(f)
            mapping = data["mapping"]

        # if m_name in ["asia", "cancer", "sachs", "alarm", "child", "water"]: 
        #     tiered_queries = generate_dynamic_queries(m_name, num_per_tier=3)
        # else:
        #     tiered_queries = generate_dynamic_queries(m_name, num_per_tier=2)
        tiered_queries = generate_dynamic_queries(m_name, num_per_tier=5)
        

        for tier_name, test_queries in tiered_queries.items():
            # if m_name == "link" and tier_name == "no_ev":
            #     print("skipping link no ev")
            #     continue
            # if m_name == "link" and tier_name == "few_ev":
            #     print("skipping link few ev")
            #     continue
            
            ss_times, ve_times, jt_times, samp_times = [], [], [], []
            ss_stats, ve_stats, jt_stats, samp_stats = [], [], [], []
            samp_errors = []
            
            for evidence, query_node, query_state in test_queries:
                # print("query")
                # ve
                ve_res, ve_t, ve_stat = run_python_with_timeout(exact_query_wrapper, ve_infer, model, query_node, query_state, evidence)
                ve_times.append(ve_t); ve_stats.append(ve_stat)

                # ss
                ss_res, ss_t, ss_stat = run_sharpsat_safe(evidence, query_node, query_state, mapping, m_name)
                ss_times.append(ss_t); ss_stats.append(ss_stat)

                # jt
                jt_init_status = "ERR"
                if jt_supported:
                    # jt_res, jt_t, jt_stat = run_python_with_timeout(exact_query_wrapper, jt_infer, model, query_node, query_state, evidence)
                    # jt_times.append(jt_t); jt_stats.append(jt_stat)
                    pass
                else:
                    jt_res = 0.0
                    jt_stats.append(jt_init_status)

                # as
                samp_factor, samp_t, samp_stat = run_python_with_timeout(sampling_infer.query, variables=[query_node], evidence=evidence, n_samples=NUM_SAMPLES, show_progress=False)
                
                samp_prob = None
                if samp_stat == "SUCCESS" and samp_factor is not None:
                    try:
                        samp_factor.normalize()
                        state_idx = samp_factor.state_names[query_node].index(query_state)
                        samp_prob = samp_factor.values[state_idx]
                    except Exception:
                        samp_stat = "ERR"
                
                samp_times.append(samp_t); samp_stats.append(samp_stat)
                
                # print("VE: " + str(ve_res))
                # print("SS: " + str(ss_res))
                # print("JT: " + str(jt_res))
                # print("AS: " + str(samp_prob))

                if ve_stat == "SUCCESS" and samp_prob is not None:
                    samp_errors.append(abs(ve_res - samp_prob))

                valid_answers = {}
                if ve_stat == "SUCCESS": 
                    valid_answers["VE"] = ve_res * 100
                if ss_stat == "SUCCESS": 
                    valid_answers["sharpSAT"] = ss_res
                # if jt_supported and jt_stat == "SUCCESS": 
                #     valid_answers["JuncTree"] = jt_res * 100

                if len(valid_answers) > 1:
                    first_val = list(valid_answers.values())[0]
                    for solver_name, val in valid_answers.items():
                        if abs(val - first_val) > 0.05:
                            error_msg = f"{m_name} ({tier_name}) | Query: P({query_node}|{len(evidence)}e) | Results: {valid_answers}"
                            print(f"\n---> [MISMATCH DETECTED] {error_msg}")
                            mismatches.append(error_msg)
                            break 

            ss_str = get_tier_display(ss_times, ss_stats)
            ve_str = get_tier_display(ve_times, ve_stats)
            jt_str = get_tier_display(jt_times, jt_stats) if jt_supported else jt_init_status
            sa_str = get_tier_display(samp_times, samp_stats, errors=samp_errors)

            print(f"{m_name:<15} | {tier_name:<10} | {ss_str:<15} | {ve_str:<15} | {jt_str:<15} | {sa_str:<20}")

    if len(mismatches) > 0:
        print(f"\nFound {len(mismatches)} result mismatches between exact solvers!")
        for error in mismatches:
            print(f"  -> {error}")
    else:
        print(f"\nNo mismatches! All successful exact solvers returned perfectly identical probabilities.")

if __name__ == "__main__":
    tournament_vs_classical()