import time
import json
import os
import subprocess
from query_nnf import solve_circuit
from query_sharpsat import query_probability_node_name
from query_ganak import query_probability_ganak

MODEL = "alarm"
SOLVERS = {
    "sharpsat": os.path.expanduser("~/sharpsat-td-main/bin/sharpSAT"),
    "ganak": "./solvers_bin/ganak",
    "d4": os.path.expanduser("~/d4/d4")
}

QUERY_NODE = 'CO' 
QUERY_STATE = 'LOW'

EVIDENCE_POOL = [
    ('HISTORY', 'FALSE'), ('CVP', 'NORMAL'), ('PCWP', 'NORMAL'),
    ('HYPOVOLEMIA', 'FALSE'), ('LVEDVOLUME', 'NORMAL'), ('LVFAILURE', 'FALSE'),
    ('STROKEVOLUME', 'NORMAL'), ('ERRLOWOUTPUT', 'FALSE'), ('HRBP', 'NORMAL'),
    ('HREKG', 'NORMAL'), ('ERRCAUTER', 'FALSE'), ('HRSAT', 'NORMAL'),
    ('INSUFFANESTH', 'FALSE'), ('ANAPHYLAXIS', 'FALSE'), ('TPR', 'NORMAL'),
    ('EXPCO2', 'NORMAL'), ('KINKEDTUBE', 'FALSE'), ('MINVOL', 'NORMAL'),
    ('FIO2', 'NORMAL'), ('PVSAT', 'NORMAL'), ('SAO2', 'NORMAL'),
    ('PAP', 'NORMAL'), ('PULMEMBOLUS', 'FALSE'), ('SHUNT', 'NORMAL'),
    ('INTUBATION', 'NORMAL'), ('PRESS', 'NORMAL'), ('DISCONNECT', 'FALSE'),
    ('MINVOLSET', 'NORMAL'), ('VENTTUBE', 'NORMAL'), ('VENTLUNG', 'NORMAL')
]


def run_scaling_test():
    print(f"Marginal Inference Scaling Test on {MODEL}")
    with open(f"temp_res/{MODEL}_data.json", "r") as f:
        data = json.load(f)
        mapping = data["mapping"]
        cpt_weights = {int(k): v for k, v in data["weights"].items()}

    # compile for d4
    print("Pre-compiling d4 circuit...")
    subprocess.run([SOLVERS["d4"], f"temp_res/{MODEL}.cnf", "-dDNNF", f"-out=temp_res/{MODEL}.nnf"], 
                   stdout=subprocess.DEVNULL)
    with open(f"temp_res/{MODEL}.nnf", "r") as f: 
        nnf_lines = f.readlines()

    scaling_results = []
    current_evidence = {}
    
    for i in range(0, len(EVIDENCE_POOL) + 1, 3):
        num_items = i
        print(f"\n--- Testing with {num_items} evidence items ---")
        print(f"Evidence: {current_evidence}")

        # d4
        start = time.perf_counter()
        p_num = solve_circuit(nnf_lines, {**current_evidence, QUERY_NODE: QUERY_STATE}, mapping, cpt_weights)
        p_den = solve_circuit(nnf_lines, current_evidence, mapping, cpt_weights)
        d4_time = time.perf_counter() - start

        # sharpsat
        start = time.perf_counter()
        query_probability_node_name(current_evidence, QUERY_NODE, QUERY_STATE, mapping, MODEL)
        ss_time = time.perf_counter() - start

        # ganak
        start = time.perf_counter()
        query_probability_ganak(current_evidence, QUERY_NODE, QUERY_STATE, mapping, MODEL)
        gn_time = time.perf_counter() - start

        scaling_results.append({
            "num_evidence": num_items,
            "d4": d4_time,
            "sharpsat": ss_time,
            "ganak": gn_time
        })

        if i < len(EVIDENCE_POOL):
            node, state = EVIDENCE_POOL[i]
            current_evidence[node] = state


    print("\n" + "="*60)
    print(f"{'Items':<10} | {'d4 (s)':<12} | {'sharpSAT (s)':<15} | {'Ganak (s)'}")
    print("-" * 60)
    for res in scaling_results:
        print(f"{res['num_evidence']:<10} | {res['d4']:<12.6f} | {res['sharpsat']:<15.4f} | {res['ganak']:.4f}")
    print("="*60)


if __name__ == "__main__":
    run_scaling_test()