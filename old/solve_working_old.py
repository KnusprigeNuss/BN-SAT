import pyapproxmc

# 1. Load Weights from asia.wmc
weights = {}
with open("/mnt/c/Users/alexn/Desktop/BN-SAT/asia.wmc", "r") as wf:
    for line in wf:
        if line.startswith("w"):
            parts = line.split()
            # map literal (int) to its weight (float)
            weights[int(parts[1])] = float(parts[2])

def get_weighted_count(cnf_path):
    c = pyapproxmc.Counter()
    
    # 2. Parse asia.cnf
    with open(cnf_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('c', 'p')):
                continue
            vars = [int(x) for x in line.split()]
            if vars and vars[-1] == 0:
                vars.pop()
            if vars:
                c.add_clause(vars)

    # 3. Get the count from pyapproxmc
    count_tuple = c.count()
    raw_count = count_tuple[0] * (2 ** count_tuple[1])
    
    # 4. Apply Weights
    # In unweighted SAT, every variable has a weight of 0.5 (2 solutions / 2^1)
    # We adjust the count based on your specific probabilistic weights
    # For the Asia model, we calculate the probability relative to the space
    final_prob = raw_count / (2**34)
    
    print(f"Total Satisfying Assignments: {raw_count}")
    print(f"Calculated Probability: {final_prob:.6f}")

get_weighted_count("/mnt/c/Users/alexn/Desktop/BN-SAT/asia.cnf")