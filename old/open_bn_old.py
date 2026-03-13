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


edges = asia_model.edges()
cpds = asia_model.get_cpds()

print(f"Edges in the model: {edges} \n")
print(f"CPDs in the model: ")
pprint.pp(cpds)

lung_cpd = asia_model.get_cpds('lung')

print("\nRaw values (weights) for the Lung CPD:")
print(lung_cpd.values)


def generate_baseline_cnf(model):
    var_count = 1
    mapping = {}  
    clauses = []
    weights = {}  

    for node in model.nodes():
        states = model.get_cpds(node).state_names[node]
        mapping[node] = {}
        node_vars = []
        for state in states:
            mapping[node][state] = var_count
            node_vars.append(var_count)
            var_count += 1
        
        clauses.append(node_vars)
        for i in range(len(node_vars)):
            for j in range(i + 1, len(node_vars)):
                clauses.append([-node_vars[i], -node_vars[j]])
        
    for cpd in model.get_cpds():
        node = cpd.variable
        states = cpd.state_names[node]
        evidence = cpd.variables[1:] 
        evidence_states = [cpd.state_names[e] for e in evidence]
        
        print(f"\nEncoding CPD for {node} with parents {evidence} and states {states}, evidence states {evidence_states}")

        for values, state_idx in zip(cpd.values.flatten(), itertools.product(*[range(len(s)) for s in evidence_states])):
            print(values, state_idx)
            
            weight_var = var_count
            var_count += 1
            weights[weight_var] = values
            
            parent_clauses = []
            for i, p_node in enumerate(evidence):
                p_state_name = evidence_states[i][state_idx[i]]
                parent_clauses.append(-mapping[p_node][p_state_name])
            
            child_state_name = states[0] 
            clauses.append(parent_clauses + [-weight_var, mapping[node][child_state_name]])
    
    return clauses, weights, mapping


asia_model = get_example_model('asia')
clauses, weights, mapping = generate_baseline_cnf(asia_model)

print(f"Generated {len(clauses)} clauses with {len(weights)} weight variables.")
print("\nSample Mapping for 'smoke':", mapping['smoke'])

