import time
import json
import os
import subprocess
import re
import warnings

warnings.filterwarnings("ignore")
from bn_to_map import generate_map_cnf
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination


TEST_CONFIGS = {
    "hepar2": {
        "evidence": {'surgery': 'present', 'alcoholism': 'present'}, 
        "map_nodes": ['Cirrhosis', 'gallstones', 'hepatotoxic']
    },
    "win95pts": {
        "evidence": {'Problem1': 'No_Output', 'Problem2': 'Too_Long'},
        "map_nodes": ['Problem3', 'Problem4', 'Problem5', 'Problem6']
    },
    "hailfinder": {
        "evidence": {'RHRatio': 'MoistMDryL', 'WindFieldPln': 'LongAnticyc'},
        "map_nodes": ['CapInScen', 'WindAloft', 'MidLLapse']
    },
    "andes": {
        "evidence": {'SNode_3': 'false', 'GOAL_2': 'true'}, 
        "map_nodes": ['TRY12', 'GOAL_48', 'TRY13']
    },
    "munin": {
        "evidence": {'L_SUR_DISP_CA': 'NO', 'L_DELT_DENERV': 'MILD'}, 
        "map_nodes": ['L_DELT_SPONT_NEUR_DISCH', 'L_DELT_SF_JITTER', 'L_DIFFN_ADM_MUDENS', 'L_ADM_MUDENS']
    }
}

SOLVERS = ["pgmpy_ve", "dpmc_native"]

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
        print("    [PGMPY] FATAL: Out of Memory Error! Induced width is too large.")
        return -1.0, "FAILED (OUT OF MEMORY)", time.perf_counter() - start
    except Exception as e:
        print(f"    [PGMPY] ERROR: {str(e)}")
        return -1.0, f"FAILED ({str(e)})", time.perf_counter() - start
        
    total_time = time.perf_counter() - start
    return prob, best_assign, total_time

def solve_native_dpmc(m_name, map_nodes, mapping):
    print("Generating Tree Decomposition and running Native SAT...")
    cnf_path = f"temp_res/{m_name}_map.cnf"
    
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
                    
    if prob == -1.0:
        best_assign = "FAILED (C++ ERROR/TIMEOUT)"
                    
    return prob, best_assign, total_time

def run_large_tournament():
    for m_name, config in TEST_CONFIGS.items():
        print(f"\n" + "=" * 90)
        print(f"Large MAP tournament: {m_name.upper()}")
        print(f"  Evidence: {config['evidence']}")
        print(f"  Target MAP Nodes: {config['map_nodes']}")
        print("=" * 90)
            
        print("Generating fresh MAP CNF and Data...")
        generate_map_cnf(
            model_name=m_name, 
            map_nodes=config['map_nodes'], 
            evidence=config['evidence'], 
            out_prefix=f"temp_res/{m_name}_map"
        )

        data_path = f"temp_res/{m_name}_map_data.json"
        with open(data_path, "r") as f:
            data = json.load(f)
            mapping = data["mapping"]

        results = {s: {"total_time": 0, "max_prob": -1.0, "best_assignment": None} for s in SOLVERS}

        p_ve, assign_ve, t_ve = solve_baseline_pgmpy(m_name, config['map_nodes'], config['evidence'])
        results["pgmpy_ve"]["total_time"] = t_ve
        results["pgmpy_ve"]["max_prob"] = p_ve
        results["pgmpy_ve"]["best_assignment"] = assign_ve

        p_dpmc, assign_dpmc, t_dpmc = solve_native_dpmc(m_name, config['map_nodes'], mapping)
        results["dpmc_native"]["total_time"] = t_dpmc
        results["dpmc_native"]["max_prob"] = p_dpmc
        results["dpmc_native"]["best_assignment"] = assign_dpmc

        print("\n" + "-" * 105)
        print(f"{'Solver Architecture':<20} | {'Total Time (s)':<15} | {'MAP Probability':<15} | {'MAP Assignment'}")
        print("-" * 105)
        for s in SOLVERS:
            r = results[s]
            assign_str = str(r['best_assignment'])
            print(f"{s:<20} | {r['total_time']:<15.4f} | {r['max_prob']:<15.8f} | {assign_str}")
        print("-" * 105 + "\n")

if __name__ == "__main__":
    run_large_tournament()