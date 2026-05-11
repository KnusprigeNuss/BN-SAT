from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination

model = get_example_model('asia')
inference = VariableElimination(model)

evidence = {'tub':'yes', 'dysp':'no'}
mpe_result = inference.map_query(
    variables=[v for v in model.nodes() if v not in evidence],
    evidence=evidence
)

print("pgmpy Exact MPE for Asia:", mpe_result)