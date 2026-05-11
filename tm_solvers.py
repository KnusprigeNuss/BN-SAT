import time
import subprocess
import os
import json
import random
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

from query_nnf import solve_circuit
from query_sharpsat import query_probability_node_name
from query_ganak import query_probability_ganak

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
    errors = 0
    for m_name in model_list:
        print(f"\nTournament for model: {m_name}")
        
        results[m_name] = {
            "d4_compile": 0,
            "no_ev": {"sharpsat": [], "ganak": [], "d4_eval": []},
            "few_ev": {"sharpsat": [], "ganak": [], "d4_eval": []},
            "much_ev": {"sharpsat": [], "ganak": [], "d4_eval": []}
        }
        
        with open(f"temp_res/{m_name}_data.json", "r") as f:
            data = json.load(f)
            mapping = data["mapping"]
            cpt_weights = {int(k): v for k, v in data["weights"].items()}
        
        # d4 compilation
        start_comp = time.perf_counter()
        subprocess.run([SOLVERS["d4"], f"temp_res/{m_name}.cnf", "-dDNNF", f"-out=temp_res/{m_name}.nnf"], stdout=subprocess.DEVNULL)
        results[m_name]["d4_compile"] = time.perf_counter() - start_comp

        print(f"Generating random queries...")
        tiered_queries = generate_dynamic_queries(m_name, num_per_tier=5)

        for tier_name, test_queries in tiered_queries.items():
            print(f"  --- Running Tier: {tier_name.upper()} ---")
            for evidence, query_node, query_state in test_queries:
                
                # ground truth
                ground_truth = get_ground_truth(m_name, query_node, evidence, query_state)
                print("sol: " + str(ground_truth))
                
                # d4
                with open(f"temp_res/{m_name}.nnf", "r") as f: 
                    lines = f.readlines()
                start_eval = time.perf_counter()
                p_num = solve_circuit(lines, {**evidence, query_node: query_state}, mapping, cpt_weights)
                p_den = solve_circuit(lines, evidence, mapping, cpt_weights)
                d4_res = p_num / p_den if p_den != 0 else 0.0
                results[m_name][tier_name]["d4_eval"].append(time.perf_counter() - start_eval)

                # sharpsat
                start_ss = time.perf_counter()
                res_sharpsat = query_probability_node_name(evidence, query_node, query_state, mapping, m_name)
                results[m_name][tier_name]["sharpsat"].append(time.perf_counter() - start_ss)
                
                # ganak
                start_ganak = time.perf_counter()
                res_ganak = query_probability_ganak(evidence, query_node, query_state, mapping, m_name)
                results[m_name][tier_name]["ganak"].append(time.perf_counter() - start_ganak)
                
                # verification
                d4_p = round(d4_res * 100, 4)
                gn_p = round(res_ganak, 4)
                ss_p = round(res_sharpsat, 4)
                gt_p = round(ground_truth * 100, 4)
                
                print(f"{gt_p}, {d4_p}, {gn_p}, {ss_p}")

                if not (d4_p == gn_p == ss_p == gt_p):
                    print(f"ERROR: Mismatch! d4: {d4_p}%, Ganak: {gn_p}%, sharpSAT: {ss_p}%, GT: {gt_p}%")
                    errors += 1
            

    print("\n" + "=" * 90)
    print(f"{'Model & Tier':<22} | {'Solver':<12} | {'Avg Query (s)':<15} | {'Compile (s)':<15}")
    print("-" * 90)
    
    tier_labels = [("no_ev", "No Ev"), ("few_ev", "Few Ev"), ("much_ev", "Much Ev")]
    
    for m in MODELS:
        d4_comp = results[m]["d4_compile"]
        for tier_key, tier_display in tier_labels:
            row_title = f"{m} ({tier_display})"
            
            num_q = len(results[m][tier_key]["sharpsat"])
            if num_q == 0: continue
            
            ss_avg = sum(results[m][tier_key]["sharpsat"]) / num_q
            gn_avg = sum(results[m][tier_key]["ganak"]) / num_q
            d4_avg = sum(results[m][tier_key]["d4_eval"]) / num_q

            print(f"{row_title:<22} | sharpSAT-td | {ss_avg:<15.4f} | {'N/A':<15}")
            print(f"{'':<22} | Ganak       | {gn_avg:<15.4f} | {'N/A':<15}")
            print(f"{'':<22} | d4 (KC)     | {d4_avg:<15.4f} | {d4_comp:<15.4f}") 
        print("-" * 90)
        
    print("Result deviations: " + str(errors))


if __name__ == "__main__":
    SOLVERS = {
        "sharpsat": os.path.expanduser("~/sharpsat-td-main/bin/sharpSAT"),
        "ganak": "./solvers_bin/ganak",
        "d4": os.path.expanduser("~/d4/d4")
    }
    MODELS = ["asia", "alarm"]  # "munin", "asia", "andes", "alarm", "hailfinder"
    results = {}
    tournament(MODELS)