import time
import subprocess
import os
import json
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination
from query_nnf import solve_circuit
from query_sharpsat import query_probability_node_name
from query_ganak import query_probability_ganak
import random
from pgmpy.sampling import BayesianModelSampling
from tm_solvers import generate_dynamic_queries


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


def tournament(model_list, errors = 0):
    for m_name in model_list:
        print(f"\nTournament solvers pruned expanded: {m_name}")
        
        results[m_name] = {
            "no_ev": {"sharpsat": []},
            "few_ev": {"sharpsat": []},
            "much_ev": {"sharpsat": []}
        }
        
        with open(f"temp_res/{m_name}_data.json", "r") as f:
            data = json.load(f)
            mapping = data["mapping"]
            cpt_weights = {int(k): v for k, v in data["weights"].items()}
        
        print(f"  Generating 5 random queries...")
        tiered_queries = generate_dynamic_queries(m_name, num_per_tier=5)

        for tier_name, test_queries in tiered_queries.items():
            print(f"  --- Running Tier: {tier_name.upper()} ---")
            for evidence, query_node, query_state in test_queries:
                ground_truth = get_ground_truth(m_name, query_node, evidence, query_state)
                print("sol: " + str(ground_truth))
                
                start_ss = time.perf_counter()
                res_sharpsat = query_probability_node_name(evidence, query_node, query_state, mapping, m_name)
                results[m_name][tier_name]["sharpsat"].append(time.perf_counter() - start_ss)
                
                ss_p = round(res_sharpsat, 4)
                gt_p = round(ground_truth * 100, 4)
                print(str(gt_p) + ", " + str(ss_p))

                if not (ss_p == gt_p):
                    print(f"ERROR: Mismatch! sharpSAT: {ss_p}%, GT: {gt_p}%")
                    errors = errors + 1
            
        
        
    print("\n" + "=" * 90)
    print(f"{'Model & Tier':<22} | {'Solver':<12} | {'Avg Query (s)':<15}")
    print("-" * 90)
    
    tier_labels = [("no_ev", "No Ev"), ("few_ev", "Few Ev"), ("much_ev", "Much Ev")]
    
    for m in MODELS:
        for tier_key, tier_display in tier_labels:
            row_title = f"{m} ({tier_display})"
            
            ss_avg = sum(results[m][tier_key]["sharpsat"]) / 5

            print(f"{row_title:<22} | sharpSAT-td | {ss_avg:<15.4f}")
        print("-" * 90)
    print("Result deviations: " + str(errors))


SOLVERS = {
    "sharpsat": os.path.expanduser("~/sharpsat-td-main/bin/sharpSAT"),
    "ganak": "./solvers_bin/ganak",
    "d4": os.path.expanduser("~/d4/d4")
}
MODELS = ["hepar2"]  # "munin" "asia", "andes" "alarm", "hailfinder",
results = {}
tournament(MODELS)