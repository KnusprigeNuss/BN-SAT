from IPython.display import Image
from pgmpy.utils import get_example_model
import itertools

MODEL = "alarm"
model = get_example_model(MODEL)
mapping = {} 
clauses = []
weights = {} 
var_count = 1
DEBUG = True

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
    
    # lock out combinations. [-1, -2], [-1, -3], [-2, -3]
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
        parent_lits = [-mapping[p][s] for p, s in zip(parents, combination)]
        
        parent_indices = [cpd.get_state_no(p, s) for p, s in zip(parents, combination)]
        
        for s_idx, state_name in enumerate(node_states):
            prob = float(cpd.values[tuple([s_idx] + parent_indices)])
            
            # optimization?
            # if prob == 0:
            #     clauses.append(parent_lits + [-mapping[node][state_name]])
            #     continue
            
            w_var = var_count
            var_count += 1
            weights[w_var] = prob
            
            child_var = mapping[node][state_name]
            
            parent_ctx = ", ".join([f"{p}({s})" for p, s in zip(parents, combination)])
            # weight_context[w_var] = f"Node: {node}({state_name}) | Parents: {parent_ctx if parents else 'None'}"
            node_info = f"{node}({state_name})"
            parent_info = parent_ctx if parents else "None"
            weight_context[w_var] = (node_info, parent_info)
            
            # forward: parents & weight -> child
            clauses.append(parent_lits + [-w_var, child_var])
            
            # backward: parent & child -> weight
            clauses.append(parent_lits + [-child_var, w_var])

            # Weight -> Parents
            # This forces the weight variable to be FALSE if any parent is 
            # not in the required state for this CPT row.
            for p_lit in parent_lits:
                clauses.append([-w_var, -p_lit])




if DEBUG:
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



def save_for_solver(clauses, weights, filename):
    all_vars = [abs(lit) for clause in clauses for lit in clause]
    num_vars = max(all_vars + list(weights.keys()))
    num_clauses = len(clauses)
    
    print(f"Writing {num_clauses} clauses and {num_vars} variables to {filename}.cnf")
    
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
            # intead of 1-prob in binary
            f.write(f"w -{var} 1.0\n")

filename = MODEL
save_for_solver(clauses, weights, filename)
print(f"\nCreated files: '{filename}.cnf' and '{filename}.wmc' in temp_res")


