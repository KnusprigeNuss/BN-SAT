# Imports
from IPython.display import Image
from pgmpy.utils import get_example_model
import itertools
import pprint


# Load the model
asia_model = get_example_model('asia')

# Visualize the network
# viz = asia_model.to_graphviz()
# viz.draw('asia.png', prog='neato')
# Image('asia.png')


# Access attributes of the model
# nodes = asia_model.nodes()
edges = asia_model.edges()
cpds = asia_model.get_cpds()

# print(f"Nodes in the model: {nodes} \n")
print(f"Edges in the model: {edges} \n")
print(f"CPDs in the model: ")
pprint.pp(cpds)

# print("--- Detailed CPD for Lung Cancer ---")
lung_cpd = asia_model.get_cpds('lung')
# print(lung_cpd)

# # 4. Show the raw values (the math your encoder will use)
# print(asia_model.get_cpds('smoke').state_names)
print("\nRaw values (weights) for the Lung CPD:")
print(lung_cpd.values)


def generate_baseline_cnf(model):
    var_count = 1
    mapping = {}  
    clauses = []
    weights = {}  

    # node states: yes/no encoding
    for node in model.nodes():
        states = model.get_cpds(node).state_names[node]
        mapping[node] = {}
        node_vars = []
        for state in states:
            mapping[node][state] = var_count
            node_vars.append(var_count)
            var_count += 1
        
        # both states in an array seperated with , means 1 or 2
        clauses.append(node_vars)
        for i in range(len(node_vars)):
            for j in range(i + 1, len(node_vars)):
                # complicated way to say the opposite. the two states are with -1 or -2
                clauses.append([-node_vars[i], -node_vars[j]])
                # creates 16 clauses for 8 nodes, each with 2 states (yes/no) to ensure only one state is true at a time
        
    # 2. Encode the CPD Tables
    for cpd in model.get_cpds():
        node = cpd.variable
        states = cpd.state_names[node]
        evidence = cpd.variables[1:] # Parent nodes
        evidence_states = [cpd.state_names[e] for e in evidence]
        
        print(f"\nEncoding CPD for {node} with parents {evidence} and states {states}, evidence states {evidence_states}")

        # We iterate through every cell in the probability table
        for values, state_idx in zip(cpd.values.flatten(), itertools.product(*[range(len(s)) for s in evidence_states])):
            print(values, state_idx)
            # This is one row of the table
            # e.g., P(Lung=Yes | Smoke=Yes) = 0.1
            
            # Create a Weight Variable for this specific probability
            weight_var = var_count
            var_count += 1
            weights[weight_var] = values
            
            # The Logic: (Parents_State & Weight_Var) -> Child_State
            # Which is: -Parent_State | -Weight_Var | Child_State
            parent_clauses = []
            for i, p_node in enumerate(evidence):
                p_state_name = evidence_states[i][state_idx[i]]
                parent_clauses.append(-mapping[p_node][p_state_name])
            
            # For each outcome state of the child
            child_state_name = states[0] # Simplification for binary
            clauses.append(parent_clauses + [-weight_var, mapping[node][child_state_name]])
    
    return clauses, weights, mapping


# general code 
asia_model = get_example_model('asia')
clauses, weights, mapping = generate_baseline_cnf(asia_model)

print(f"Generated {len(clauses)} clauses with {len(weights)} weight variables.")
print("\nSample Mapping for 'smoke':", mapping['smoke'])

