import time
import sys
import os
import subprocess
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pgmpy.utils import get_example_model
from bn_to_mpe_sat import generate_mpe_encoding

sys.setrecursionlimit(20000)

MODELS = ["asia", "alarm", "hailfinder"]
EVIDENCE = {
    'asia': {'tub': 'yes', 'dysp': 'no'}, 
    'alarm': {'MINVOL': 'NORMAL', 'BP': 'LOW'},
    'hailfinder': {'TempDis': 'None', 'CombClouds': 'PC'}
}

UWRMAXSAT_BIN = os.path.expanduser("~/uwrmaxsat/build/release/bin/uwrmaxsat")

def get_fast_joint_prob(model, state):
    prob = 1.0
    for cpd in model.get_cpds():
        var = cpd.variable
        var_idx = cpd.name_to_no[var][state[var]]
        
        if not cpd.variables[1:]:
            p = cpd.values[var_idx]
        else:
            parent_vars = cpd.variables[1:]
            parent_indices = tuple(cpd.name_to_no[p][state[p]] for p in parent_vars)
            idx = (var_idx,) + parent_indices
            p = cpd.values[idx]
            
        prob *= p
        if prob == 0.0:
            return 0.0
    return prob

def get_assignment_from_model(model, mapping):
    assignment = {}
    for node, states in mapping.items():
        for state, sat_var in states.items():
            if sat_var in model:
                assignment[node] = state
    return assignment

def run_uwrmaxsat(wcnf_file, mapping):
    start_time = time.perf_counter()
    try:
        result = subprocess.run([UWRMAXSAT_BIN, wcnf_file], capture_output=True, text=True, timeout=120)
        solve_time = time.perf_counter() - start_time
        
        sat_model = []
        is_optimal = False
        
        for line in result.stdout.split('\n'):
            if line.startswith('s OPTIMUM FOUND'):
                is_optimal = True
            elif line.startswith('v '):
                lits = line.split()[1:]
                for lit in lits:
                    sat_model.append(int(lit))
                    
        if is_optimal and sat_model:
            assignment = get_assignment_from_model(sat_model, mapping)
            return solve_time, assignment
        else:
            print("UWRMAXSAT Failed to find optimum or parse output.")
            return -1.0, {}
            
    except subprocess.TimeoutExpired:
        print("UWRMAXSAT TIMEOUT")
        return -1.0, {}
    except Exception as e:
        print(f"Error {e}")
        return -1.0, {}

def run_maxsat_tournament():
    for m_name in MODELS:
        print(f"\nMPE TOURNAMENT: RC2 vs UWrMaxSat - {m_name.upper()}")
        model = get_example_model(m_name)
        
        print("Generating MaxSAT (WCNF) encoding...")
        mapping, _ = generate_mpe_encoding(model, EVIDENCE[m_name], m_name)
        wcnf_file = f"temp_res/{m_name}_mpe.wcnf"
        
        if m_name == "hailfinder":
            # taking too long
            rc2_time = -1.0
            rc2_prob = 0.0
        else:
            rc2_time, rc2_prob = 0, 0.0
            try:
                start_rc2 = time.perf_counter()
                wcnf = WCNF(from_file=wcnf_file)
                with RC2(wcnf) as rc2:
                    rc2_model = rc2.compute()
                    rc2_time = time.perf_counter() - start_rc2
                    if rc2_model:
                        assign = get_assignment_from_model(rc2_model, mapping)
                        rc2_prob = get_fast_joint_prob(model, {**EVIDENCE[m_name], **assign})
            except (KeyboardInterrupt, Exception):
                print("  [RC2 INTERRUPTED]")
                rc2_time = -1.0

        uwr_time, uwr_assign = run_uwrmaxsat(wcnf_file, mapping)
        
        uwr_prob = 0.0
        if uwr_time >= 0:
            uwr_prob = get_fast_joint_prob(model, {**EVIDENCE[m_name], **uwr_assign})


        print("-" * 65)
        print(f"{'MaxSAT Architecture':<25} | {'Time (s)':<15} | {'MPE Probability'}")
        print("-" * 65)
        
        rc2_t_str = f"{rc2_time:<15.4f}" if rc2_time >= 0 else "SKIPPED / OOM"
        rc2_p_str = f"{rc2_prob:<15.6e}" if rc2_time >= 0 else "N/A"
        print(f"{'RC2 (PySAT Python)':<25} | {rc2_t_str} | {rc2_p_str}")
        
        uwr_t_str = f"{uwr_time:<15.4f}" if uwr_time >= 0 else "TIMEOUT"
        uwr_p_str = f"{uwr_prob:<15.6e}" if uwr_time >= 0 else "N/A"
        print(f"{'UWrMaxSat (Native C++)':<25} | {uwr_t_str} | {uwr_p_str}")
        print("-" * 65)

if __name__ == "__main__":
    run_maxsat_tournament()