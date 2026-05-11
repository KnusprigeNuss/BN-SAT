import time
import sys
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination, BeliefPropagation

from bn_to_mpe_sat import generate_mpe_encoding

sys.setrecursionlimit(20000)

MODELS = ["asia", "alarm", "hailfinder"]
EVIDENCE = {
    'asia': {'tub': 'yes', 'dysp': 'no'}, 
    'alarm': {'MINVOL': 'NORMAL', 'BP': 'LOW'},
    'hailfinder': {'TempDis': 'None', 'CombClouds': 'PC'}
}

def get_assignment_from_model(model, mapping):
    assignment = {}
    for node, states in mapping.items():
        for state, sat_var in states.items():
            if sat_var in model:
                assignment[node] = state
    return assignment

def run_mpe_classical_tournament():
    for m_name in MODELS:
        print(f"\n🏆 MPE TOURNAMENT: RC2 vs Classical - {m_name.upper()}")
        model = get_example_model(m_name)
        mpe_nodes = [n for n in model.nodes() if n not in EVIDENCE[m_name]]
        
        mapping, weights = generate_mpe_encoding(model, EVIDENCE[m_name], m_name)
        wcnf_file = f"temp_res/{m_name}_mpe.wcnf"
        
        print("start rc2")
        start_rc2 = time.perf_counter()
        wcnf = WCNF(from_file=wcnf_file)
        with RC2(wcnf) as rc2:
            rc2_model = rc2.compute()
            rc2_time = time.perf_counter() - start_rc2
            rc2_assignment = get_assignment_from_model(rc2_model, mapping)
        print("end rc2")


        print("start VE")
        ve = VariableElimination(model)
        start_ve = time.perf_counter()
        ve_assignment = ve.map_query(variables=mpe_nodes, evidence=EVIDENCE[m_name], show_progress=False)
        ve_time = time.perf_counter() - start_ve
        print("end VE")

        print("start JT")
        bp = BeliefPropagation(model)
        start_bp = time.perf_counter()
        try:
            bp_assignment = bp.map_query(variables=mpe_nodes, evidence=EVIDENCE[m_name], show_progress=False)
            bp_time = time.perf_counter() - start_bp
        except Exception as e:
            bp_time = -1.0
            bp_assignment = {}
        print("end JT")


        ve_match = all(rc2_assignment.get(n) == ve_assignment.get(n) for n in mpe_nodes)
        bp_match = all(rc2_assignment.get(n) == bp_assignment.get(n) for n in mpe_nodes) if bp_time != -1.0 else False
        # ve_match = True
        # bp_match = True
        # ve_time = 0.0

        print("-" * 65)
        print(f"{'Solver':<20} | {'Time (s)':<12} | {'Matches RC2?'}")
        print("-" * 65)
        print(f"{'RC2 (MaxSAT)':<20} | {rc2_time:<12.4f} | Base Truth")
        print(f"{'Var. Elimination':<20} | {ve_time:<12.4f} | {'YES' if ve_match else 'NO'}")
        
        if bp_time == -1.0:
             print(f"{'Junction Tree':<20} | {'FAILED':<12} | N/A")
        else:
             print(f"{'Junction Tree':<20} | {bp_time:<12.4f} | {'YES' if bp_match else 'NO'}")
        print("-" * 65)

if __name__ == "__main__":
    run_mpe_classical_tournament()