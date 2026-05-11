import time
import json
import os
import random
import warnings
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

from bn_to_cnf_wmc_w_pruning import generate_wmc_cnf
from query_sharpsat import get_joint_wmc_node_name

warnings.filterwarnings("ignore")

MODELS = ["pathfinder"] #"pathfinder"
NUM_PER_TIER = 5 

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


def solve_baseline_pgmpy(m_name, target_state, evidence):
    model = get_example_model(m_name)
    inference = VariableElimination(model)
    
    start = time.perf_counter()
    try:
        query_vars = list(target_state.keys())
        if evidence:
            cond_factor = inference.query(variables=query_vars, evidence=evidence, show_progress=False)
            cond_factor.reduce([(k, target_state[k]) for k in query_vars])
            p_q_given_e = float(cond_factor.values.item() if hasattr(cond_factor.values, 'item') else cond_factor.values)
            
            ev_vars = list(evidence.keys())
            ev_factor = inference.query(variables=ev_vars, show_progress=False)
            ev_factor.reduce([(k, evidence[k]) for k in ev_vars])
            p_e = float(ev_factor.values.item() if hasattr(ev_factor.values, 'item') else ev_factor.values)
            
            prob = p_q_given_e * p_e
        else:
            factor = inference.query(variables=query_vars, show_progress=False)
            factor.reduce([(k, target_state[k]) for k in query_vars])
            prob = float(factor.values.item() if hasattr(factor.values, 'item') else factor.values)
            
    except MemoryError:
        return -1.0, time.perf_counter() - start 
    except Exception as e:
        return -1.0, time.perf_counter() - start
        
    return prob, time.perf_counter() - start


def run_pruning_tournament():
    results = {}
    for m_name in MODELS:
        print(f"\nTournament VE and SharpSat with pruning: {m_name}")
        results[m_name] = {
            "no_ev": {"sharpsat": [], "ve": []},
            "few_ev": {"sharpsat": [], "ve": []},
            "much_ev": {"sharpsat": [], "ve": []},
            "max_tree": {"sharpsat": [], "ve": []}
        }
        
        print(f"Generating mathematically valid queries...")
        tiered_queries = generate_dynamic_queries_with_maxtree(m_name, num_per_tier=NUM_PER_TIER)

        for tier_name, test_queries in tiered_queries.items():
            print(f"  --- Running Tier: {tier_name.upper()} ---")
            
            for evidence, query_node, query_state in test_queries:
                target_state = {query_node: query_state}
                out_prefix = f"temp_res/{m_name}" 
                
                cnf_path, wmc_path, data_path = generate_wmc_cnf(
                    model_name=m_name, 
                    query_nodes=[query_node], 
                    evidence=evidence, 
                    out_prefix=out_prefix
                )
                
                with open(data_path, "r") as f:
                    mapping = json.load(f)["mapping"]

                # print("start sharpsat")
                start_ss = time.perf_counter()
                prob_ss = get_joint_wmc_node_name(target_state, mapping, m_name)
                print(prob_ss)
                results[m_name][tier_name]["sharpsat"].append(time.perf_counter() - start_ss)
                # print("end sharpsat")

                # print("start VE")
                # prob_ve, time_ve = solve_baseline_pgmpy(m_name, target_state, evidence)
                # results[m_name][tier_name]["ve"].append(time_ve if prob_ve != -1.0 else -1.0)
                # print("end VE")



    print("\n" + "=" * 80)
    print(f"{'Model & Tier':<25} | {'Solver':<20} | {'Avg Time (s)':<15}")
    print("-" * 80)
    
    tier_labels = [("no_ev", "Empty"), 
                   ("few_ev", "Few Ev"), 
                   ("much_ev", "Much Ev"), 
                   ("max_tree", "Max Tree")]
    
    for m in MODELS:
        for tier_key, tier_display in tier_labels:
            row_title = f"{m} ({tier_display})"
            
            ss_times = results[m][tier_key]["sharpsat"]
            ve_times = results[m][tier_key]["ve"]
            
            ss_avg = sum(ss_times) / len(ss_times) if ss_times else 0.0
            
            valid_ve = [t for t in ve_times if t != -1.0]
            if len(valid_ve) > 0:
                ve_avg = f"{sum(valid_ve) / len(valid_ve):<15.4f}"
            else:
                ve_avg = "OOM / CRASHED  "

            print(f"{row_title:<25} | {'sharpSAT-td':<20} | {ss_avg:<15.4f}")
            print(f"{'':<25} | {'Var. Elimination':<20} | {ve_avg}")
        print("-" * 80)

if __name__ == "__main__":
    run_pruning_tournament()