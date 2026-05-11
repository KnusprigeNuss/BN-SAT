import time
import sys
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pysat.examples.lsu import LSU
from pgmpy.utils import get_example_model

from bn_to_mpe_sat import generate_mpe_encoding

sys.setrecursionlimit(20000)

MODELS = ["asia", "alarm", "hailfinder"]
EVIDENCE = {
    'asia': {'tub': 'yes', 'dysp': 'no'}, 
    'alarm': {'MINVOL': 'NORMAL', 'BP': 'LOW'},
    'hailfinder': {'TempDis': 'None', 'CombClouds': 'PC'}
}

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

def run_maxsat_tournament():
    for m_name in MODELS:
        print(f"\n🏆 MPE TOURNAMENT: MaxSAT Paradigms - {m_name.upper()}")
        model = get_example_model(m_name)
        
        print("  Generating MaxSAT (WCNF) encoding...")
        mapping, _ = generate_mpe_encoding(model, EVIDENCE[m_name], m_name)
        wcnf_file = f"temp_res/{m_name}_mpe.wcnf"
        
        if m_name == "hailfinder":
            print("  [WARNING] Hailfinder is a massive combinatorial space.")
            print("  (Press CTRL+C if a solver hangs to skip to the next one!)")

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
            print("  [RC2 INTERRUPTED BY USER]")
            rc2_time = -1.0

        # 2. LSU (Linear Search MaxSAT)
        lsu_time, lsu_prob = 0, 0.0
        try:
            start_lsu = time.perf_counter()
            wcnf = WCNF(from_file=wcnf_file)
            with LSU(wcnf, expect_interrupt=True) as lsu:
                lsu_model = lsu.compute()
                lsu_time = time.perf_counter() - start_lsu
                if lsu_model:
                    assign = get_assignment_from_model(lsu_model, mapping)
                    lsu_prob = get_fast_joint_prob(model, {**EVIDENCE[m_name], **assign})
        except (KeyboardInterrupt, Exception):
            print("  [LSU INTERRUPTED BY USER]")
            lsu_time = -1.0

        print("-" * 65)
        print(f"{'MaxSAT Algorithm':<25} | {'Time (s)':<15} | {'MPE Probability'}")
        print("-" * 65)
        
        rc2_t_str = f"{rc2_time:<15.4f}" if rc2_time >= 0 else "TIMEOUT"
        rc2_p_str = f"{rc2_prob:<15.6e}" if rc2_time >= 0 else "N/A"
        print(f"{'RC2 (Core-Guided)':<25} | {rc2_t_str} | {rc2_p_str}")
        
        lsu_t_str = f"{lsu_time:<15.4f}" if lsu_time >= 0 else "TIMEOUT"
        lsu_p_str = f"{lsu_prob:<15.6e}" if lsu_time >= 0 else "N/A"
        print(f"{'LSU (Linear Search)':<25} | {lsu_t_str} | {lsu_p_str}")
        print("-" * 65)

if __name__ == "__main__":
    run_maxsat_tournament()