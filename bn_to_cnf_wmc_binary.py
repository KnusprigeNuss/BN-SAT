from IPython.display import Image
from pgmpy.utils import get_example_model
import itertools
import pprint

MODEL = "asia"
asia_model = get_example_model(MODEL)
mapping = {} 
clauses = []
weights = {} 
var_count = 1
DEBUG = True

# clauses for node states (Note: only yes/no) 
for node in asia_model.nodes():
    mapping[node] = {'yes': var_count, 'no': var_count + 1}
    clauses.append([var_count, var_count + 1])
    clauses.append([-var_count, -(var_count + 1)])
    var_count += 2

weight_context = {}

for cpd in asia_model.get_cpds():
    node = cpd.variable
    parents = cpd.variables[1:]

    parent_states = ["yes", "no"]
    state_combinations = list(itertools.product(parent_states, repeat=len(parents)))

    # print(node)
    # print(state_combinations)

    for combination in state_combinations:
        indices = [0] + [0 if s == 'yes' else 1 for s in combination]

        # accessing the correct cpd value by going through the indices iteratively
        prob = float(cpd.values[tuple(indices)])

        w_var = var_count
        var_count += 1
        weights[w_var] = prob

        parent_ctx = ", ".join([f"{p}({s})" for p, s in zip(parents, combination)])
        weight_context[w_var] = f"Node: {node} | Parents: {parent_ctx if parents else 'None'}"

        parent_literals = [-mapping[p_name][p_state] for p_name, p_state in zip(parents, combination)]
        child_yes = mapping[node]['yes']

        # forward declaration and equivalence
        clauses.append(parent_literals + [-w_var, child_yes])
        clauses.append(parent_literals + [-child_yes, w_var])



print(f"Generated {len(clauses)} clauses.")
print(f"Used {var_count-1} variables with {var_count-1-len(weights)} being variables for node states and {len(weights)} variables being weight variables.")

if DEBUG:
    print("\n--- NODE TO SAT VARIABLE MAPPING ---")
    print(f"{'Node Name':<15} | {'Yes Var':<8} | {'No Var':<8}")
    print("-" * 35)
    for node, vars in mapping.items():
        print(f"{node:<15} | {vars['yes']:<8} | {vars['no']:<8}")


    print("\n--- DETAILED WEIGHT VARIABLE MAPPING ---")
    print(f"{'Weight Var':<12} | {'Prob':<6} | {'Context'}")
    print("-" * 70)
    for w_var, ctx in weight_context.items():
        print(f"{w_var:<12} | {weights[w_var]:<6} | {ctx}")


def save_for_solver(clauses, weights, filename):
    all_vars = []
    for clause in clauses:
        for x in clause:
            all_vars.append(abs(x))
    
    num_vars = max(all_vars + list(weights.keys()))
    num_clauses = len(clauses)
    
    print(f"Writing {num_clauses} clauses and {num_vars} variables to {filename}.cnf")
    
    with open(f"temp_res/{filename}.cnf", "w") as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")
    with open(f"temp_res/{filename}.wmc", "w") as f:
        for var, prob in weights.items():
            f.write(f"w {var} {prob}\n")
            f.write(f"w -{var} {1.0 - prob}\n")


save_for_solver(clauses, weights, MODEL)
print(f"\nCreated files: '{MODEL}.cnf' and '{MODEL}.wmc' in temp_res")


