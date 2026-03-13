from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination

model = get_example_model('alarm')
inference = VariableElimination(model)

result = inference.query(
    variables=['HISTORY'], 
    evidence={}
)

print(result)

# result = inference.query(
#     variables=['INTUBATION'], 
#     evidence={'SHUNT': 'NORMAL', 'PRESS': 'ZERO'}
# )

# print(result)

# result = inference.query(
#     variables=['VENTMACH'], 
#     evidence={}
# )

# state_idx = model.get_cpds('VENTMACH').get_state_no('VENTMACH', 'NORMAL')
# prob_normal = result.values[state_idx]

# print(f"P(VENTMACH=NORMAL) = {prob_normal:.4f}")

# print(result)