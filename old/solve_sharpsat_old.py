import subprocess
import re

SHARPSAT_PATH = "/home/nieale4/sharpSAT/build/sharpSAT"
BASE_CNF = "/mnt/c/Users/alexn/Desktop/BN-SAT/asia.cnf"

def get_count(additional_clauses=[]):
    temp_cnf = "temp_query.cnf"
    
    # 1. Read existing CNF
    with open(BASE_CNF, "r") as f:
        lines = f.readlines()
    
    # 2. Update Header and Write Temp File
    with open(temp_cnf, "w") as f:
        for line in lines:
            if line.startswith('p cnf'):
                parts = line.split()
                new_clauses = int(parts[3]) + len(additional_clauses)
                f.write(f"p cnf {parts[2]} {new_clauses}\n")
            else:
                f.write(line)
        for clause in additional_clauses:
            f.write(f"{clause} 0\n")
    
    # 3. Run SharpSAT
    result = subprocess.run([SHARPSAT_PATH, temp_cnf], capture_output=True, text=True)
    
    # Modified Part 4 in solve.py
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        if "# solutions" in line:
            # Look at the next line for the actual number
            next_line = lines[i+1].strip()
            match = re.search(r'(\d+)', next_line)
            if match:
                return int(match.group(1))
    
    print(f"DEBUG: SharpSAT output was: {result.stdout}")
    return 0

# Variables: Smoking=30, LungCancer=31
print("Calculating Denominator: P(Smoking=True)...")
count_e = get_count([5]) 
print(f"-> Count: {count_e}")

print("Calculating Numerator: P(LungCancer=True AND Smoking=True)...")
count_q_e = get_count([5, 9]) 
print(f"-> Count: {count_q_e}")

if count_e > 0:
    probability = count_q_e / count_e
    print(f"\n--- SUCCESS ---")
    print(f"P(LungCancer | Smoking=True) = {probability:.6f}")
else:
    print("\n--- ERROR ---")
    print("Could not find a valid solution count. Check DEBUG output above.")
