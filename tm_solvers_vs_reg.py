import time
import json
import os
import random
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination, BeliefPropagation, ApproxInference
from pgmpy.sampling import BayesianModelSampling
from query_sharpsat import query_probability_node_name
from tm_solvers import generate_dynamic_queries


def tournament_vs_classical():
    MODELS = ["hepar2"]  # "hailfinder" "asia", "alarm"
    NUM_SAMPLES = 2000
    results = {}

    for m_name in MODELS:
        print(f"\nTournament WMC vs REG: {m_name}")
        
        results[m_name] = {
            "no_ev": {"sharpsat": [], "ve": [], "jt": [], "sampling": []},
            "few_ev": {"sharpsat": [], "ve": [], "jt": [], "sampling": []},
            "much_ev": {"sharpsat": [], "ve": [], "jt": [], "sampling": []}
        }
        
        model = get_example_model(m_name)
        ve_infer = VariableElimination(model)
        sampling_infer = ApproxInference(model)
        
        if m_name != "hailfinder" and m_name != "hepar2":
            jt_infer = BeliefPropagation(model)

        with open(f"temp_res/{m_name}_data.json", "r") as f:
            data = json.load(f)
            mapping = data["mapping"]

        print(f"  Generating 5 random queries...")
        tiered_queries = generate_dynamic_queries(m_name, num_per_tier=5)

        for tier_name, test_queries in tiered_queries.items():
            print(f"  --- Running Tier: {tier_name.upper()} ---")
            for evidence, query_node, query_state in test_queries:
                
                # sharpsat
                # print("starting sharpsat")
                start = time.perf_counter()
                query_probability_node_name(evidence, query_node, query_state, mapping, m_name)
                results[m_name][tier_name]["sharpsat"].append(time.perf_counter() - start)
                # print("finished sharpsat")


                # ve
                # print("starting VE")
                start = time.perf_counter()
                ve_infer.query(variables=[query_node], evidence=evidence, show_progress=False)
                results[m_name][tier_name]["ve"].append(time.perf_counter() - start)
                # print("finishing VE")
                


                # jt
                # print("starting JT")
                if m_name != "hailfinder" and m_name != "hepar2":
                    start = time.perf_counter()
                    jt_infer.query(variables=[query_node], evidence=evidence, show_progress=False)
                    results[m_name][tier_name]["jt"].append(time.perf_counter() - start)
                else:
                    results[m_name][tier_name]["jt"].append(0.0)
                # print("finishing VE")
                

                # sampling
                # print("starting sampling")
                # if m_name != "hailfinder":
                start = time.perf_counter()
                sampling_infer.query(
                    variables=[query_node], 
                    evidence=evidence, 
                    n_samples=NUM_SAMPLES, 
                    show_progress=False
                )
                results[m_name][tier_name]["sampling"].append(time.perf_counter() - start)
            # else:
                results[m_name][tier_name]["sampling"].append(0.0)
                # print("finishing sampling")
                



    print("\n" + "="*95)
    print(f"{'Model & Tier':<22} | {'Inference Method':<22} | {'Avg Query (s)':<15} | {'Type'}")
    print("-" * 95)
    
    tier_labels = [("no_ev", "No Ev"), ("few_ev", "Few Ev"), ("much_ev", "Much Ev")]
    
    for m in MODELS:
        for tier_key, tier_display in tier_labels:
            row_title = f"{m} ({tier_display})"
            
            num_q = len(results[m][tier_key]["sharpsat"])
            if num_q == 0: continue
            
            ss_avg = sum(results[m][tier_key]["sharpsat"]) / num_q
            ve_avg = sum(results[m][tier_key]["ve"]) / num_q
            jt_avg = sum(results[m][tier_key]["jt"]) / num_q
            samp_avg = sum(results[m][tier_key]["sampling"]) / num_q

            jt_display = f"{jt_avg:<15.4f}" if m != "hailfinder" else f"{'N/A':<15}"

            print(f"{row_title:<22} | sharpSAT-td (WMC)    | {ss_avg:<15.4f} | Logic-Based")
            print(f"{'':<22} | Var. Elimination     | {ve_avg:<15.4f} | Classical Exact")
            print(f"{'':<22} | Junction Tree        | {jt_display} | Classical Exact")
            print(f"{'':<22} | Approx. Sampling     | {samp_avg:<15.4f} | Approximate")
        print("-" * 95)

if __name__ == "__main__":
    tournament_vs_classical()