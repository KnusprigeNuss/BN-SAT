from IPython.display import Image
from pgmpy.utils import get_example_model
import itertools
import pprint

asia_model = get_example_model('asia')
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

# encoding the derivations (Note: only for max. 2 parents per node)
for cpd in asia_model.get_cpds():
    node = cpd.variable
    parents = cpd.variables[1:]

    # roots
    if len(parents) == 0:
        p_yes = cpd.values[0]
        w_var = var_count
        var_count += 1
        weights[w_var] = p_yes
        weight_context[w_var] = f"Node: {node} | Parent: None"
        
        # if weight -> node must be true and the equivalence clause
        clauses.append([-w_var, mapping[node]['yes']])
        clauses.append([-mapping[node]['yes'], w_var])

    # one parent
    elif len(parents) == 1:
        p_name = parents[0]
        for p_state in ['yes', 'no']:
            prob = cpd.values[0][0 if p_state == 'yes' else 1]
            w_var = var_count
            var_count += 1
            weights[w_var] = prob
            weight_context[w_var] = f"Node: {node} | Parent: {p_name}({p_state})"
            
            p_var = mapping[p_name][p_state]
            c_yes = mapping[node]['yes']
            
            # (parent & weight) -> child
            clauses.append([-p_var, -w_var, c_yes])
            # equivalence: (parent & child) -> weight
            clauses.append([-p_var, -c_yes, w_var])

    # two parents
    elif len(parents) == 2:
        p1 = parents[0]
        p2 = parents[1]
        combos = [('yes', 'yes'), ('yes', 'no'), ('no', 'yes'), ('no', 'no')]
        for s1, s2 in combos:
            idx1 = (0 if s1 == 'yes' else 1)
            idx2 = (0 if s2 == 'yes' else 1)
            prob = cpd.values[0][idx1][idx2]
            w_var = var_count
            var_count += 1
            weights[w_var] = prob
            weight_context[w_var] = f"Node: {node} | Parents: {p1}({s1}), {p2}({s2})"
            
            p1_var = mapping[p1][s1]
            p2_var = mapping[p2][s2]
            c_yes = mapping[node]['yes']
            
            # (p1 & p2 & weight) -> child
            clauses.append([-p1_var, -p2_var, -w_var, c_yes])
            # equivalence: (p1 & p2 & child) -> weight
            clauses.append([-p1_var, -p2_var, -c_yes, w_var])



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

filename = "asia"
save_for_solver(clauses, weights, filename)
print(f"\nCreated files: '{filename}.cnf' and '{filename}.wmc' in temp_res")


