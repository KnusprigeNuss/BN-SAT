from IPython.display import Image
from pgmpy.utils import get_example_model
import itertools
import json
from pgmpy.readwrite import BIFReader
import os

# MODEL = "net_parents_5"
# reader = BIFReader(path=f"synthetic_networks/{MODEL}.bif")
# model = reader.get_model()

def generate_wmc_cnf(model, model_name, debug=True):
    mapping = {} 
    clauses = []
    weights = {} 
    var_count = 1
    weight_context = {}

    for node in model.nodes():
        cpd = model.get_cpds(node)
        states = cpd.state_names[node]
        mapping[node] = {}
        
        state_vars = []
        for state in states:
            mapping[node][state] = var_count
            state_vars.append(var_count)
            var_count += 1
            
        # [1, 2, 3], like before
        clauses.append(state_vars)
        
        # lock out combinations of two. [-1, -2], [-1, -3], [-2, -3]
        for pair in itertools.combinations(state_vars, 2):
            clauses.append([-pair[0], -pair[1]])


    for cpd in model.get_cpds():
        node = cpd.variable
        parents = cpd.variables[1:]
        node_states = cpd.state_names[node]
        # print(node_states)

        parent_state_lists = [cpd.state_names[p] for p in parents]
        # print(parent_state_lists)
        state_combinations = list(itertools.product(*parent_state_lists))
        # print(node)
        # print(state_combinations)


        for combination in state_combinations:
            # print(cpd)
            # combinations of parents and states with their SAT var. 
            parent_lits = [-mapping[p][s] for p, s in zip(parents, combination)]
            # print(parent_lits)

            parent_state_indices = [cpd.get_state_no(p, s) for p, s in zip(parents, combination)]
            
            for state_idx, state_name in enumerate(node_states):
                # fetch cpt value
                prob = float(cpd.values[tuple([state_idx] + parent_state_indices)])
                
                # optimization?
                # if prob == 0:
                #     clauses.append(parent_lits + [-mapping[node][state_name]])
                #     continue

                w_var = var_count
                var_count += 1
                weights[w_var] = prob
                
                child_var = mapping[node][state_name]
                
                # for debugging
                parent_ctx = ", ".join([f"{p}({s})" for p, s in zip(parents, combination)])
                node_info = f"{node}({state_name})"
                parent_info = parent_ctx if parents else "None"
                weight_context[w_var] = (node_info, parent_info)
                
                # forward: parents & weight -> child
                clauses.append(parent_lits + [-w_var, child_var])
                
                # backward: parent & child -> weight
                clauses.append(parent_lits + [-child_var, w_var])

                # weight -> parents
                # weight must be false if any parent is not in correct state according to CPT
                for p_lit in parent_lits:
                    clauses.append([-w_var, -p_lit])



    if debug:
        print("\n--- NODE TO SAT VARIABLE MAPPING ---")
        print(f"{'Node Name':<20} | {'State Name':<15} | {'SAT Var'}")
        print("-" * 50)
        for node, states_dict in mapping.items():
            for state_name, sat_var in states_dict.items():
                print(f"{node:<20} | {state_name:<15} | {sat_var}")
            print("-" * 50)

        print("\n--- DETAILED WEIGHT VARIABLE MAPPING ---")
        print(f"{'Weight Var':<12} | {'Prob':<8} | {'Node':<30} | {'Parents'}")
        print("-" * 100)
        for w_var, (node_info, parent_info) in weight_context.items():
            prob_val = weights[w_var]
            print(f"{w_var:<12} | {prob_val:<8.4f} | {node_info:<30} | {parent_info}")

        print(f"Generated {len(clauses)} clauses.")
        print(f"Used {var_count-1} variables with {var_count-1-len(weights)} being variables for node states and {len(weights)} variables being weight variables.")

    save_for_solver(clauses, weights, model_name, mapping)
    if debug:
        print(f"Created files: '{model_name}.cnf', '{model_name}.wmc', and '{model_name}_data.json'")



def save_for_solver(clauses, weights, filename, mapping):
    all_vars = [abs(lit) for clause in clauses for lit in clause]
    num_vars = max(all_vars + list(weights.keys()))
    num_clauses = len(clauses)
    
    # print(f"Writing {num_clauses} clauses and {num_vars} variables to {filename}.cnf")
    
    # write cnf in dimacs format
    with open(f"temp_res/{filename}.cnf", "w") as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")
            
    # weight map
    with open(f"temp_res/{filename}.wmc", "w") as f:
        for var, prob in weights.items():
            f.write(f"w {var} {prob:.15f}\n")
            # if weight not true the variable shouldnt change anything -> 1.0
            f.write(f"w -{var} 1.0\n")
    
    data_bundle = {
        "mapping": mapping,
        "weights": weights,
        "num_vars": num_vars,
        "clauses": clauses
    }
    
    with open(f"temp_res/{filename}_data.json", "w") as f:
        json.dump(data_bundle, f)

# filename = MODEL
# save_for_solver(clauses, weights, filename)
# print(f"\nCreated files: '{filename}.cnf' and '{filename}.wmc' and '{filename}_data.json' in temp_res")

# MODELS = ["cancer", "earthquake", "asia", "sachs", "child", "insurance", "alarm", "hepar2", "hailfinder", "win95pts"]
if __name__ == "__main__":
    MODEL_NAME = "alarm"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")

    MODEL_NAME = "child"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")

    MODEL_NAME = "water"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")

    MODEL_NAME = "barley"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")

    MODEL_NAME = "hailfinder"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")

    MODEL_NAME = "win95pts"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")

    MODEL_NAME = "andes"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")

    MODEL_NAME = "link"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")

    MODEL_NAME = "pathfinder"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")

    MODEL_NAME = "munin2"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")

    MODEL_NAME = "munin"
    test_model = get_example_model(MODEL_NAME)
    generate_wmc_cnf(test_model, MODEL_NAME, debug=False)
    print(f"\nTest {MODEL_NAME} completed successfully!")