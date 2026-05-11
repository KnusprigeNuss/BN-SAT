import time
import json
import os
import subprocess
import re
import warnings
import random
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

from bn_to_map import generate_map_cnf

warnings.filterwarnings("ignore")

MODELS = ["hailfinder"] 
NUM_PER_TIER = 1 

def generate_dynamic_map_queries(model_name, num_per_tier=3):
    model = get_example_model(model_name)
    nodes = list(model.nodes())
    total_nodes = len(nodes)
    
    few_ev_count = max(1, int(total_nodes * 0.10))
    much_ev_count = max(2, int(total_nodes * 0.35))
    
    small_map_count = max(2, int(total_nodes * 0.05))
    large_map_count = max(4, int(total_nodes * 0.20))
    
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=num_per_tier, show_progress=False)
    
    queries = {"small_map_no_ev": [], "small_map_much_ev": [], "large_map_few_ev": []}
    
    for i in range(num_per_tier):
        sample = samples.iloc[i].to_dict()
        shuffled_nodes = nodes.copy()
        random.shuffle(shuffled_nodes)
        
        # 1. SMALL MAP, NO EV
        # map_1 = shuffled_nodes[:small_map_count]
        # queries["small_map_no_ev"].append(({}, map_1))
        
        # 2. SMALL MAP, MUCH EV
        # map_2 = shuffled_nodes[:small_map_count]
        # rem_2 = [n for n in shuffled_nodes if n not in map_2]
        # ev_2 = {n: sample[n] for n in rem_2[:much_ev_count]}
        # queries["small_map_much_ev"].append((ev_2, map_2))
        
        # 3. LARGE MAP, FEW EV
        map_3 = shuffled_nodes[:large_map_count]
        rem_3 = [n for n in shuffled_nodes if n not in map_3]
        ev_3 = {n: sample[n] for n in rem_3[:few_ev_count]}
        queries["large_map_few_ev"].append((ev_3, map_3))
        
    return queries


def solve_baseline_pgmpy(m_name, map_nodes, evidence):
    model = get_example_model(m_name)
    inference = VariableElimination(model)
    
    start = time.perf_counter()
    try:
        best_assign = inference.map_query(variables=map_nodes, evidence=evidence, show_progress=False)
        p_map_given_e = 1.0
        if map_nodes:
            cond_factor = inference.query(variables=map_nodes, evidence=evidence, show_progress=False)
            cond_factor.reduce([(k, best_assign[k]) for k in map_nodes])
            p_map_given_e = float(cond_factor.values.item() if hasattr(cond_factor.values, 'item') else cond_factor.values)
            
        p_e = 1.0
        if evidence:
            ev_vars = list(evidence.keys())
            ev_factor = inference.query(variables=ev_vars, show_progress=False)
            ev_factor.reduce([(k, evidence[k]) for k in ev_vars])
            p_e = float(ev_factor.values.item() if hasattr(ev_factor.values, 'item') else ev_factor.values)
            
        prob = p_map_given_e * p_e
    except MemoryError:
        return -1.0, {}, time.perf_counter() - start
    except Exception as e:
        return -1.0, {}, time.perf_counter() - start
        
    return prob, best_assign, time.perf_counter() - start


def solve_native_dpmc(m_name, map_nodes, mapping):
    cnf_path = f"temp_res/{m_name}_map.cnf"
    
    subprocess.run(f"sed -i 's/0\\.000000000000000/0.00000000000000000000001/g' {cnf_path}", shell=True)
    subprocess.run(f"sed -i 's/ 0\\.0 / 0.00000000000000000000001 /g' {cnf_path}", shell=True)
    
    cmd = (
        f'./solvers_bin/DPMC/bin/lg "./solvers_bin/DPMC/bin/flow_cutter_pace17 -p 100" < {cnf_path} | '
        f'./solvers_bin/DPMC/bin/dmc --cf={cnf_path} --wc=1 --pc=1 --dp=c --er=1 --mf=2'
    )
    
    start = time.perf_counter()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    total_time = time.perf_counter() - start
    
    prob = -1.0
    best_assign = {}
    
    prob_match = re.search(r"exact double prec-sci\s+([eE\d\.-]+)", result.stdout)
    if prob_match: 
        prob = float(prob_match.group(1))
        
    v_match = re.search(r"^v\s+(.*)$", result.stdout, re.MULTILINE)
    if v_match:
        lits = v_match.group(1).split()
        rev_map = {v: (n, s) for n, states in mapping.items() for s, v in states.items()}
        for lit in lits:
            var = abs(int(lit))
            if var in rev_map:
                node, state = rev_map[var]
                if node in map_nodes and int(lit) > 0:
                    best_assign[node] = state
                    
    return prob, best_assign, total_time


def run_dynamic_map_tournament():
    results = {}
    
    for m_name in MODELS:
        print(f"\nMap tournament tiers: {m_name.upper()}")
        results[m_name] = {
            "small_map_no_ev": {"dpmc": [], "ve": []},
            "small_map_much_ev": {"dpmc": [], "ve": []},
            "large_map_few_ev": {"dpmc": [], "ve": []}
        }

        print(f"Generating random MAP scenarios...")
        tiered_queries = generate_dynamic_map_queries(m_name, num_per_tier=NUM_PER_TIER)

        for tier_name, test_queries in tiered_queries.items():
            print(f"  --- Running Tier: {tier_name.upper()} ---")
            for evidence, map_nodes in test_queries:
                out_prefix = f"temp_res/{m_name}_map"
                
                generate_map_cnf(model_name=m_name, map_nodes=map_nodes, evidence=evidence, out_prefix=out_prefix)
                
                with open(f"{out_prefix}_data.json", "r") as f:
                    mapping = json.load(f)["mapping"]

                # 2. dpmc
                p_dpmc, assign_dpmc, t_dpmc = solve_native_dpmc(m_name, map_nodes, mapping)
                results[m_name][tier_name]["dpmc"].append(t_dpmc)

                # 3. ve
                p_ve, assign_ve, t_ve = solve_baseline_pgmpy(m_name, map_nodes, evidence)
                results[m_name][tier_name]["ve"].append(t_ve if p_ve != -1.0 else -1.0)


    print("\n" + "=" * 80)
    print(f"{'Model & Tier':<35} | {'Solver':<20} | {'Avg Time (s)':<15}")
    print("-" * 80)
    
    tier_labels = [
        ("small_map_no_ev", "Small MAP, No Ev"), 
        ("small_map_much_ev", "Small MAP, Much Ev"), 
        ("large_map_few_ev", "Large MAP, Few Ev")
    ]
    
    for m in MODELS:
        for tier_key, tier_display in tier_labels:
            row_title = f"{m} ({tier_display})"
            
            dpmc_times = results[m][tier_key]["dpmc"]
            ve_times = results[m][tier_key]["ve"]
            
            dpmc_avg = sum(dpmc_times) / len(dpmc_times) if dpmc_times else 0.0
            
            valid_ve = [t for t in ve_times if t != -1.0]
            if len(valid_ve) > 0:
                ve_avg = f"{sum(valid_ve) / len(valid_ve):<15.4f}"
            else:
                ve_avg = "OOM / CRASHED  "

            print(f"{row_title:<35} | {'DPMC Native':<20} | {dpmc_avg:<15.4f}")
            print(f"{'':<35} | {'Var. Elimination':<20} | {ve_avg}")
        print("-" * 80)

if __name__ == "__main__":
    run_dynamic_map_tournament()