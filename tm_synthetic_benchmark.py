import os
import time
import subprocess
import json
import matplotlib.pyplot as plt
from pgmpy.readwrite import BIFReader
from bn_to_cnf_wmc import generate_wmc_cnf
from query_sharpsat import create_wcnf_file


def run_sharpsat_td(file_name):
    abs_file_path = os.path.abspath(file_name)
    solver_dir = os.path.dirname(os.path.expanduser("~/sharpsat-td-main/bin/sharpSAT"))

    cmd = ["./sharpSAT", "-WE", "-decot", "0.001", "-tmpdir", ".", "-prec", "10", abs_file_path]
    
    start_time = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=solver_dir, timeout=120)
        solve_time = time.perf_counter() - start_time
        
        if "exact arb float" in result.stdout:
            return solve_time
        else:
            return -1.0 
    except subprocess.TimeoutExpired:
        return -1.0 
    except Exception as e:
        print(f"Error running sharpSAT-td: {e}")
        return -1.0


def benchmark_category(category_name, param_values, file_template):
    results = []
    print(f"\n--- Running Benchmarks: Scaling {category_name} ---")
    
    safe_cat = category_name.replace(" ", "_").replace("(", "").replace(")", "")
    
    for val in param_values:
        bif_file = f"synthetic_networks/{file_template.format(val)}"
        if not os.path.exists(bif_file):
            print(f"  Error: Missing {bif_file}, skipping testcase...")
            results.append((val, 0.0, 0))
            continue
            
        print(f"  Testing {category_name} = {val}...", end="", flush=True)
        
        reader = BIFReader(bif_file)
        model = reader.get_model()
        model_name = f"synth_{safe_cat}_{val}"
        
        generate_wmc_cnf(model, model_name=model_name, debug=False)
        
        first_node = list(model.nodes())[0]
        with open(f"temp_res/{model_name}_data.json", "r") as f:
            data_json = json.load(f)
            mapping = data_json["mapping"]
            num_clauses = len(data_json["clauses"])
        
        first_state = list(mapping[first_node].keys())[0]
        query_sat_var = mapping[first_node][first_state]
        
        wcnf_file = f"temp_res/{model_name}_query.wcnf"
        create_wcnf_file(wcnf_file, [query_sat_var], model_name)
        
        t_solve = run_sharpsat_td(wcnf_file)
        
        if t_solve >= 0:
            print(f" Done in {t_solve:.4f}s | Clauses: {num_clauses}")
        else:
            print(f" TIMEOUT (or OOM) | Clauses: {num_clauses}")
            
        results.append((val, t_solve, num_clauses))
    return results


def plot_dual_axis(ax, title, xlabel, x_time, y_time, x_cl, y_cl):
        color1 = 'tab:blue'
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Execution Time (s)', color=color1, fontweight='bold')
        line1, = ax.plot(x_time, y_time, marker='o', color=color1, linewidth=2, label='Solve Time (s)')
        ax.tick_params(axis='y', labelcolor=color1)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Number of Clauses', color=color2, fontweight='bold')
        line2, = ax2.plot(x_cl, y_cl, marker='s', linestyle='--', color=color2, linewidth=2, label='Total Clauses')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax.legend(handles=[line1, line2], loc='upper left')


def run_benchmarks():
    node_counts = [10, 20, 50, 70, 90, 95, 100]
    parent_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    state_counts = [2, 3, 4, 5, 6, 7, 8]

    res_nodes = benchmark_category("Nodes", node_counts, "net_nodes_{}.bif")
    res_parents = benchmark_category("Max Parents (Treewidth)", parent_counts, "net_parents_{}.bif")
    res_states = benchmark_category("States per Node", state_counts, "net_states_{}.bif")

    print("\n" + "="*80)
    print(f"{'Network Attribute':<25} | {'Parameter Value':<15} | {'sharpSAT Time (s)':<18} | {'Clauses'}")
    print("="*80)
    
    for val, t, c in res_nodes:
        time_str = f"{t:.4f}" if t >= 0 else "TIMEOUT"
        print(f"{'Total Nodes':<25} | {val:<15} | {time_str:<18} | {c}")
    print("-" * 80)
    for val, t, c in res_parents:
        time_str = f"{t:.4f}" if t >= 0 else "TIMEOUT"
        print(f"{'Max Parents (Density)':<25} | {val:<15} | {time_str:<18} | {c}")
    print("-" * 80)
    for val, t, c in res_states:
        time_str = f"{t:.4f}" if t >= 0 else "TIMEOUT"
        print(f"{'States per Node':<25} | {val:<15} | {time_str:<18} | {c}")
    print("="*80)


    print("\nGenerating dual-axis plots...")
    plot_n_x_time = [v for v, t, c in res_nodes if t >= 0]
    plot_n_y_time = [t for v, t, c in res_nodes if t >= 0]
    plot_n_x_cl   = [v for v, t, c in res_nodes]
    plot_n_y_cl   = [c for v, t, c in res_nodes]
    
    plot_p_x_time = [v for v, t, c in res_parents if t >= 0]
    plot_p_y_time = [t for v, t, c in res_parents if t >= 0]
    plot_p_x_cl   = [v for v, t, c in res_parents]
    plot_p_y_cl   = [c for v, t, c in res_parents]
    
    plot_s_x_time = [v for v, t, c in res_states if t >= 0]
    plot_s_y_time = [t for v, t, c in res_states if t >= 0]
    plot_s_x_cl   = [v for v, t, c in res_states]
    plot_s_y_cl   = [c for v, t, c in res_states]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    

    plot_dual_axis(axs[0], 'Scaling Network Size', 'Number of Nodes', 
                   plot_n_x_time, plot_n_y_time, plot_n_x_cl, plot_n_y_cl)
    plot_dual_axis(axs[1], 'Scaling Density', 'Max Parents per Node', 
                   plot_p_x_time, plot_p_y_time, plot_p_x_cl, plot_p_y_cl)
    plot_dual_axis(axs[2], 'Scaling State Space', 'States per Node', 
                   plot_s_x_time, plot_s_y_time, plot_s_x_cl, plot_s_y_cl)


    plt.tight_layout()
    plt.savefig('scalability_results.png', dpi=300)
    print("Saved plot as 'scalability_results.png'!")

if __name__ == "__main__":
    run_benchmarks()