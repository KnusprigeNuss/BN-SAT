import matplotlib.pyplot as plt

def plot_dual_axis(ax, title, xlabel, x_time, y_time, x_cl, y_cl):
    color1 = 'tab:blue'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Execution Time (s)', color=color1, fontweight='bold', fontsize=10)
    line1, = ax.plot(x_time, y_time, marker='o', color=color1, linewidth=2, label='Solve Time (s)')
    ax.tick_params(axis='y', labelcolor=color1)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Number of Clauses', color=color2, fontweight='bold', fontsize=10)
    line2, = ax2.plot(x_cl, y_cl, marker='s', linestyle='--', color=color2, linewidth=2, label='Total Clauses')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax.legend(handles=[line1, line2], loc='upper left', framealpha=0.9)


def generate_plots():
    # results taken from: tm_synthetic_benchmark.py
    res_nodes = [
        (10, 0.0064, 232),
        (20, 0.0095, 500),
        (50, 0.0881, 1556),
        (70, 0.4826, 2208),
        (90, 34.9750, 2892),
        (100, 55.5572, 3200),
        (105, 230.3759, 3398),
        (110, -1.0, 3664)  
    ]

    res_parents = [
        (1, 0.0078, 412),
        (2, 0.0175, 932),
        (3, 0.0836, 2260),
        (4, 0.7271, 4956),
        (5, 2.7777, 11144),
        (6, 3.5884, 23468),
        (7, 19.3003, 50032),
        (8, 100.2304, 110768),
        (9, 290.8656, 208164),
        (10, -1.0, 460828)  
    ]

    res_states = [
        (2, 0.0166, 868),
        (3, 0.0780, 2688),
        (4, 5.4603, 6562),
        (5, 15.1316, 12585),
        (6, 29.8131, 22332),
        (7, -1.0, 35429),  
        (8, -1.0, 44902),  
        (9, -1.0, 71427)   
    ]

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


    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[1, 1:3])
    axs = [ax1, ax2, ax3]
    

    plot_dual_axis(axs[0], 'Scaling Network Size', 'Number of Nodes (n)', 
                   plot_n_x_time, plot_n_y_time, plot_n_x_cl, plot_n_y_cl)
    

    plot_dual_axis(axs[1], 'Scaling Density', 'Max Parents per Node (k)', 
                   plot_p_x_time, plot_p_y_time, plot_p_x_cl, plot_p_y_cl)
    

    plot_dual_axis(axs[2], 'Scaling State Space', 'States per Node (s)', 
                   plot_s_x_time, plot_s_y_time, plot_s_x_cl, plot_s_y_cl)

    plt.tight_layout(pad=3.0)
    
    output_filename = 'scalability_results_cust1.png'
    plt.savefig(output_filename, dpi=300)
    print(f"SUCCESS: Publication-ready figure generated and exported to '{output_filename}'!")

if __name__ == "__main__":
    generate_plots()