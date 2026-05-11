import random
import itertools
import os

def generate_random_bif(filename, num_nodes, max_parents=2, states_per_node=2, edge_prob=0.3):
    nodes = [f"v{i}" for i in range(num_nodes)]
    parents = {n: [] for n in nodes}
    
    for i in range(1, num_nodes):
        possible_parents = nodes[:i]
        random.shuffle(possible_parents)
        num_p = 0
        for p in possible_parents:
            if random.random() < edge_prob and num_p < max_parents:
                parents[nodes[i]].append(p)
                num_p += 1

    state_names = [f"s{i}" for i in range(states_per_node)]
    states_str = ", ".join(state_names)

    with open(filename, "w") as f:
        f.write(f"network synthetic_{num_nodes}_nodes {{\n}}\n")
        
        for n in nodes:
            f.write(f"variable {n} {{\n")
            f.write(f"  type discrete [ {states_per_node} ] {{ {states_str} }};\n")
            f.write("}\n")
            
        for n in nodes:
            pa = parents[n]
            if not pa:
                f.write(f"probability ( {n} ) {{\n")
                probs = [random.random() for _ in range(states_per_node)]
                s = sum(probs)
                probs = [p/s for p in probs]
                f.write("  table " + ", ".join(f"{p:.6f}" for p in probs) + ";\n")
                f.write("}\n")
            else:
                f.write(f"probability ( {n} | {', '.join(pa)} ) {{\n")
                parent_combinations = list(itertools.product(state_names, repeat=len(pa)))
                for combo in parent_combinations:
                    probs = [random.random() for _ in range(states_per_node)]
                    s = sum(probs)
                    probs = [p/s for p in probs]
                    combo_str = ", ".join(combo)
                    f.write(f"  ({combo_str}) " + ", ".join(f"{p:.6f}" for p in probs) + ";\n")
                f.write("}\n")
    print(f"Generated {filename} | Nodes: {num_nodes} | Max Parents: {max_parents} | States: {states_per_node}")


if __name__ == "__main__":
    if not os.path.exists("synthetic_networks"):
        os.makedirs("synthetic_networks")

    for n in [10, 20, 50, 70, 90, 100]:
        generate_random_bif(f"synthetic_networks/net_nodes_{n}.bif", num_nodes=n, max_parents=2, states_per_node=2)

    for p in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        generate_random_bif(f"synthetic_networks/net_parents_{p}.bif", num_nodes=30, max_parents=p, states_per_node=2, edge_prob=0.8)

    for s in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        generate_random_bif(f"synthetic_networks/net_states_{s}.bif", num_nodes=30, max_parents=2, states_per_node=s)