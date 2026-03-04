# Imports
from IPython.display import Image
from pgmpy.utils import get_example_model
import itertools
import pprint

# Load the model
asia_model = get_example_model('asia')
mapping = {} # (Node, State) -> Int
clauses = []
weights = {} # Int -> Prob
var_count = 1

# 1. Map all nodes to SAT variables (1=Yes, 2=No, 3=Yes, 4=No...)
for node in asia_model.nodes():
    mapping[node] = {'yes': var_count, 'no': var_count + 1}
    # Constraint: Must be Yes OR No, but not both
    clauses.append([var_count, var_count + 1])
    clauses.append([-var_count, -(var_count + 1)])
    var_count += 2

# 2. Encode the Logic
for cpd in asia_model.get_cpds():
    node = cpd.variable
    parents = cpd.variables[1:]

    print(f"\nEncoding CPD for {node} with parents {parents}...")

    # --- ROOT NODES (0 Parents) ---
    if len(parents) == 0:
        p_yes = cpd.values[0]
        w_var = var_count; var_count += 1
        weights[w_var] = p_yes
        clauses.append([-w_var, mapping[node]['yes']])
        print(f"Root node '{node}': P(Yes)={p_yes} -> Weight Var={w_var}")

    # --- SINGLE PARENT NODES (1 Parent) ---
    elif len(parents) == 1:
        p_name = parents[0]
        # Rules: If Parent=Yes -> Child=Yes (prob1), If Parent=No -> Child=Yes (prob2)
        for p_state in ['yes', 'no']:
            prob = cpd.values[0][0 if p_state == 'yes' else 1]
            w_var = var_count; var_count += 1
            weights[w_var] = prob
            # SAT: (-Parent_State | -Weight | Child_Yes)
            clauses.append([-mapping[p_name][p_state], -w_var, mapping[node]['yes']])

    # --- TWO PARENT NODES (e.g., "either") ---
    elif len(parents) == 2:
        p1, p2 = parents[0], parents[1]
        # There are 4 combinations: (Y,Y), (Y,N), (N,Y), (N,N)
        combos = [('yes', 'yes'), ('yes', 'no'), ('no', 'yes'), ('no', 'no')]
        
        for i, (s1, s2) in enumerate(combos):
            # pgmpy indexing for 2 parents: cpd.values[outcome][p1_idx][p2_idx]
            idx1 = 0 if s1 == 'yes' else 1
            idx2 = 0 if s2 == 'yes' else 1
            prob = cpd.values[0][idx1][idx2]
            
            w_var = var_count; var_count += 1
            weights[w_var] = prob
            
            # SAT: (-P1_State | -P2_State | -Weight | Child_Yes)
            clauses.append([-mapping[p1][s1], -mapping[p2][s2], -w_var, mapping[node]['yes']])




print(f"Generated {len(clauses)} clauses with {len(weights)} weight variables.")
print("\nSample Mapping for 'smoke':", mapping['smoke'])

