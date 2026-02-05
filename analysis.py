# =============================================================================
# STRUCTURAL ANALYSIS FOR AIR TRAFFIC NETWORK
# =============================================================================
# This code implements three "fancy" structural measures for academic analysis:
# 1. Structural Sensitivity Analysis (Brexit Check)
# 2. Degree Assortativity Time-Series (Hub-and-Spoke Analysis)
# 3. Weighted vs. Binary Correlation
# =============================================================================

import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# 1. STRUCTURAL SENSITIVITY ANALYSIS (BREXIT CHECK)
# -----------------------------------------------------------------------------
# Purpose: Quantify how much structural change is a "data artifact" from Brexit
# versus actual traffic shifts by removing UK/GB nodes and measuring impact.

def create_ghost_graph(G, prefixes_to_remove=('UK_', 'GB_')):
    """
    Create a 'Ghost Graph' by removing all nodes that start with specified prefixes.
    
    Parameters:
    -----------
    G : nx.DiGraph
        Original graph
    prefixes_to_remove : tuple
        Node name prefixes to remove (default: UK and GB airports)
    
    Returns:
    --------
    nx.DiGraph
        Ghost graph with specified nodes removed
    """
    G_ghost = G.copy()
    nodes_to_remove = [node for node in G_ghost.nodes() 
                       if any(node.startswith(prefix) for prefix in prefixes_to_remove)]
    G_ghost.remove_nodes_from(nodes_to_remove)
    return G_ghost, nodes_to_remove

def get_largest_scc_size(G):
    """
    Get the size of the largest strongly connected component.
    
    Parameters:
    -----------
    G : nx.DiGraph
        Directed graph
    
    Returns:
    --------
    int
        Number of nodes in the largest SCC
    """
    if G.number_of_nodes() == 0:
        return 0
    sccs = list(nx.strongly_connected_components(G))
    if not sccs:
        return 0
    return max(len(scc) for scc in sccs)

def structural_sensitivity_analysis(graphs, base_year=2014):
    """
    Perform structural sensitivity analysis by comparing original and ghost graphs.
    
    Parameters:
    -----------
    graphs : dict
        Dictionary of NetworkX DiGraph objects keyed by "G_{year}"
    base_year : int
        Year to analyze (default: 2014)
    
    Returns:
    --------
    dict
        Results containing original metrics, ghost metrics, and percentage changes
    """
    G_key = f"G_{base_year}"
    if G_key not in graphs:
        raise KeyError(f"Graph for year {base_year} not found in graphs dictionary")
    
    G_original = graphs[G_key]
    G_ghost, removed_nodes = create_ghost_graph(G_original)
    
    # Calculate metrics for original graph
    original_density = nx.density(G_original)
    original_clustering = nx.average_clustering(G_original.to_undirected())
    original_scc_size = get_largest_scc_size(G_original)
    
    # Calculate metrics for ghost graph
    ghost_density = nx.density(G_ghost)
    ghost_clustering = nx.average_clustering(G_ghost.to_undirected()) if G_ghost.number_of_nodes() > 0 else 0
    ghost_scc_size = get_largest_scc_size(G_ghost)
    
    # Calculate percentage changes
    def pct_change(original, modified):
        if original == 0:
            return 0.0
        return ((modified - original) / original) * 100
    
    results = {
        'year': base_year,
        'removed_nodes': removed_nodes,
        'num_removed_nodes': len(removed_nodes),
        'original': {
            'nodes': G_original.number_of_nodes(),
            'edges': G_original.number_of_edges(),
            'density': original_density,
            'avg_clustering': original_clustering,
            'largest_scc_size': original_scc_size
        },
        'ghost': {
            'nodes': G_ghost.number_of_nodes(),
            'edges': G_ghost.number_of_edges(),
            'density': ghost_density,
            'avg_clustering': ghost_clustering,
            'largest_scc_size': ghost_scc_size
        },
        'pct_change': {
            'nodes': pct_change(G_original.number_of_nodes(), G_ghost.number_of_nodes()),
            'edges': pct_change(G_original.number_of_edges(), G_ghost.number_of_edges()),
            'density': pct_change(original_density, ghost_density),
            'avg_clustering': pct_change(original_clustering, ghost_clustering),
            'largest_scc_size': pct_change(original_scc_size, ghost_scc_size)
        }
    }
    
    return results

# -----------------------------------------------------------------------------
# 2. DEGREE ASSORTATIVITY TIME-SERIES (HUB-AND-SPOKE ANALYSIS)
# -----------------------------------------------------------------------------
# Purpose: Measure network structure evolution over time.
# Interpretation:
# - Negative values: Hub-and-Spoke model (high-degree nodes connect to low-degree)
# - Values near zero: Shift toward decentralized Point-to-Point travel
# - Positive values: Assortative mixing (high connects to high)

def calculate_assortativity_timeseries(graphs):
    """
    Calculate degree assortativity coefficient for each year.
    
    Parameters:
    -----------
    graphs : dict
        Dictionary of NetworkX DiGraph objects keyed by "G_{year}"
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with year and assortativity coefficient
    """
    results = []
    
    for key in sorted(graphs.keys()):
        year = int(key.split('_')[1])
        G = graphs[key]
        
        try:
            # Calculate degree assortativity (for in+out degree in directed graphs)
            assortativity = nx.degree_assortativity_coefficient(G)
        except nx.NetworkXError:
            assortativity = np.nan
        
        results.append({
            'year': year,
            'assortativity': assortativity,
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges()
        })
    
    return pd.DataFrame(results)

def plot_assortativity_timeseries(assortativity_df, save_path=None):
    """
    Plot the assortativity time-series with interpretation guide.
    
    Parameters:
    -----------
    assortativity_df : pd.DataFrame
        DataFrame from calculate_assortativity_timeseries
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the assortativity values
    ax.plot(assortativity_df['year'], assortativity_df['assortativity'], 
            'o-', linewidth=2, markersize=8, color='steelblue', label='Degree Assortativity')
    
    # Add horizontal reference lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1, label='Neutral (r=0)')
    
    # Shade regions for interpretation
    ax.axhspan(-1, 0, alpha=0.1, color='red', label='Hub-and-Spoke (Disassortative)')
    ax.axhspan(0, 1, alpha=0.1, color='green', label='Point-to-Point (Assortative)')
    
    # Labels and formatting
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Degree Assortativity Coefficient (r)', fontsize=12)
    ax.set_title('Degree Assortativity Time-Series:\nHub-and-Spoke vs. Point-to-Point Network Structure', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(assortativity_df['year'].min() - 0.5, assortativity_df['year'].max() + 0.5)
    ax.set_ylim(-0.5, 0.2)  # Adjust based on your data
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key values
    for idx, row in assortativity_df.iterrows():
        ax.annotate(f'{row["assortativity"]:.3f}', 
                    xy=(row['year'], row['assortativity']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig

# -----------------------------------------------------------------------------
# 3. WEIGHTED VS. BINARY CORRELATION
# -----------------------------------------------------------------------------
# Purpose: Analyze whether "having many routes" correlates with "having many passengers"
# Goal: Show if adding new routes at small airports shifts systemic power

def calculate_degree_strength_correlation(G, weight='weight'):
    """
    Calculate correlation between binary degree and weighted node strength.
    
    Parameters:
    -----------
    G : nx.DiGraph
        NetworkX directed graph with weighted edges
    weight : str
        Edge attribute name for weights
    
    Returns:
    --------
    dict
        Dictionary with degree/strength data and correlation statistics
    """
    node_data = []
    
    for node in G.nodes():
        # Binary degree (count of connections)
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)
        total_degree = in_degree + out_degree
        
        # Weighted strength (sum of passenger weights)
        in_strength = sum(d.get(weight, 1) for u, v, d in G.in_edges(node, data=True))
        out_strength = sum(d.get(weight, 1) for u, v, d in G.out_edges(node, data=True))
        total_strength = in_strength + out_strength
        
        node_data.append({
            'node': node,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_degree': total_degree,
            'in_strength': in_strength,
            'out_strength': out_strength,
            'total_strength': total_strength
        })
    
    df = pd.DataFrame(node_data)
    
    # Calculate Pearson correlation
    if len(df) > 2:
        r_total, p_total = pearsonr(df['total_degree'], df['total_strength'])
        r_in, p_in = pearsonr(df['in_degree'], df['in_strength'])
        r_out, p_out = pearsonr(df['out_degree'], df['out_strength'])
    else:
        r_total, p_total = np.nan, np.nan
        r_in, p_in = np.nan, np.nan
        r_out, p_out = np.nan, np.nan
    
    return {
        'node_data': df,
        'correlation': {
            'total': {'r': r_total, 'p': p_total},
            'in': {'r': r_in, 'p': p_in},
            'out': {'r': r_out, 'p': p_out}
        }
    }

def weighted_binary_correlation_analysis(graphs, years=[2014, 2024]):
    """
    Analyze weighted vs. binary correlation for specified years.
    
    Parameters:
    -----------
    graphs : dict
        Dictionary of NetworkX DiGraph objects keyed by "G_{year}"
    years : list
        Years to analyze
    
    Returns:
    --------
    dict
        Results for each year with correlation data
    """
    results = {}
    
    for year in years:
        G_key = f"G_{year}"
        if G_key not in graphs:
            print(f"Warning: Graph for year {year} not found, skipping...")
            continue
        
        G = graphs[G_key]
        analysis = calculate_degree_strength_correlation(G)
        
        results[year] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'node_data': analysis['node_data'],
            'correlation': analysis['correlation']
        }
    
    return results

def plot_degree_strength_correlation(results, save_path=None):
    """
    Plot degree vs. strength scatter plots for comparison.
    
    Parameters:
    -----------
    results : dict
        Results from weighted_binary_correlation_analysis
    save_path : str, optional
        Path to save the figure
    """
    years = sorted(results.keys())
    n_years = len(years)
    
    fig, axes = plt.subplots(1, n_years, figsize=(7 * n_years, 6))
    if n_years == 1:
        axes = [axes]
    
    for ax, year in zip(axes, years):
        df = results[year]['node_data']
        r = results[year]['correlation']['total']['r']
        p = results[year]['correlation']['total']['p']
        
        # Scatter plot with log scale for better visualization
        ax.scatter(df['total_degree'], df['total_strength'], alpha=0.6, s=30, c='steelblue')
        
        # Add trend line
        if not np.isnan(r):
            z = np.polyfit(df['total_degree'], df['total_strength'], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(df['total_degree'].min(), df['total_degree'].max(), 100)
            ax.plot(x_line, p_line(x_line), 'r--', alpha=0.8, linewidth=2, label=f'Trend (r={r:.3f})')
        
        ax.set_xlabel('Binary Degree (Number of Destinations)', fontsize=11)
        ax.set_ylabel('Node Strength (Sum of Passenger Weights)', fontsize=11)
        ax.set_title(f'{year}\nr = {r:.4f}, p = {p:.2e}\nNodes = {len(df)}', 
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to log scale for better visualization of range
        ax.set_yscale('log')
    
    plt.suptitle('Weighted vs. Binary Correlation Analysis:\nDoes Having Many Routes Mean Having Many Passengers?', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig

# -----------------------------------------------------------------------------
# 4. FORMAL SMALL-WORLDNESS COEFFICIENT (ω) ANALYSIS - L06 METHODOLOGY
# -----------------------------------------------------------------------------
# Purpose: Prove mathematically if the European air space functions as a 
# Small-World network by comparing against Erdős-Rényi random graphs.
#
# L06 Formula (from L06.pdf slide 43): ω = Lr/L - C/Cl
#   - ω ≈ 0: Small-World network (between random and lattice)
#   - ω > 0: More random-like (closer to ER random graph)
#   - ω < 0: More lattice-like (closer to regular lattice)
#
# where:
#   - L = Average shortest path length of real network
#   - L_rand = Average shortest path length of Erdős-Rényi random graph
#   - C = Clustering coefficient of real network
#   - C_reg = Clustering coefficient of a regular lattice (baseline)
#
# For a ring lattice with k neighbors: C_reg ≈ 3(k-2) / 4(k-1) ≈ 0.75 for large k

def calculate_lattice_clustering(n, avg_degree):
    """
    Calculate the expected clustering coefficient for a regular ring lattice.
    
    For a ring lattice with k neighbors on each side (total degree 2k):
    C_lattice = 3(k-2) / 4(k-1) for k >= 2
    
    Parameters:
    -----------
    n : int
        Number of nodes
    avg_degree : float
        Average degree of the network
    
    Returns:
    --------
    float
        Expected clustering coefficient for equivalent lattice
    """
    k = avg_degree / 2  # k = number of neighbors on each side
    if k >= 2:
        c_lattice = (3 * (k - 2)) / (4 * (k - 1))
        return max(c_lattice, 0.0)  # Ensure non-negative
    else:
        # For very sparse graphs, use a simpler approximation
        return 0.75  # Theoretical max for lattice


def calculate_small_worldness_l06(G, n_random_graphs=10, seed=42):
    """
    Calculate Small-Worldness coefficient (ω) following L06 methodology.
    
    Formula: ω = Lr/L - C/Cl 
    
    Parameters:
    -----------
    G : nx.DiGraph or nx.Graph
        NetworkX graph to analyze
    n_random_graphs : int
        Number of random graphs to generate for averaging (default: 10)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary with all small-world metrics
    """
    np.random.seed(seed)
    
    # Convert to undirected for small-world analysis
    G_undirected = G.to_undirected() if G.is_directed() else G.copy()
    
    n = G_undirected.number_of_nodes()
    m = G_undirected.number_of_edges()
    avg_degree = 2 * m / n if n > 0 else 0
    
    # Calculate probability p for Erdős-Rényi graph with same expected edges
    p = (2 * m) / (n * (n - 1)) if n > 1 else 0
    
    # Real network metrics
    C_real = nx.average_clustering(G_undirected)
    
    # Lattice baseline clustering coefficient
    C_reg = calculate_lattice_clustering(n, avg_degree)
    
    # Get largest connected component for path length calculation
    if nx.is_connected(G_undirected):
        G_lcc = G_undirected
    else:
        lcc = max(nx.connected_components(G_undirected), key=len)
        G_lcc = G_undirected.subgraph(lcc).copy()
    
    L_real = nx.average_shortest_path_length(G_lcc)
    lcc_size = G_lcc.number_of_nodes()
    lcc_coverage = lcc_size / n * 100
    
    # Generate random graphs and compute average metrics
    L_rand_values = []
    C_rand_values = []
    
    for i in range(n_random_graphs):
        # Generate Erdős-Rényi random graph with same n and m
        G_rand = nx.gnm_random_graph(n, m, seed=seed + i)
        
        # Clustering coefficient
        C_rand_values.append(nx.average_clustering(G_rand))
        
        # Path length (use largest connected component)
        if nx.is_connected(G_rand):
            G_rand_lcc = G_rand
        else:
            components = list(nx.connected_components(G_rand))
            if components:
                lcc_rand = max(components, key=len)
                G_rand_lcc = G_rand.subgraph(lcc_rand).copy()
            else:
                continue
        
        if G_rand_lcc.number_of_nodes() > 1:
            L_rand_values.append(nx.average_shortest_path_length(G_rand_lcc))
    
    # Average random graph metrics
    L_rand = np.mean(L_rand_values) if L_rand_values else np.nan
    C_rand = np.mean(C_rand_values) if C_rand_values else np.nan
    
    # Calculate Small-Worldness coefficient (ω) 
    # ω = Lr/L - C/Cl 
    # Lr = random graph path length, L = real network path length
    # C = real clustering, Cl = lattice clustering
    if L_rand > 0 and L_real > 0 and C_reg > 0:
        L_ratio = L_rand / L_real      # Lr/L
        C_ratio_omega = C_real / C_reg  # C/Cl
        omega = L_ratio - C_ratio_omega
        
        # Alternative sigma coefficient (Humphries & Gurney): σ = (C/C_rand) / (L/L_rand)
        # σ > 1 indicates small-world
        sigma = (C_real / C_rand) / (L_real / L_rand) if C_rand > 0 else np.nan
    else:
        omega = np.nan
        sigma = np.nan
        L_ratio = np.nan
        C_ratio_omega = np.nan
    
    # Interpretation based on ω
    if not np.isnan(omega):
        if -0.5 <= omega <= 0.5:
            interpretation = "SMALL-WORLD NETWORK"
        elif omega > 0.5:
            interpretation = "RANDOM-LIKE NETWORK"
        else:
            interpretation = "LATTICE-LIKE NETWORK"
    else:
        interpretation = "UNABLE TO COMPUTE"
    
    return {
        'n': n,
        'm': m,
        'avg_degree': avg_degree,
        'p': p,
        'C_real': C_real,
        'C_rand': C_rand,
        'C_reg': C_reg,
        'C_ratio': C_real / C_rand if C_rand > 0 else np.nan,
        'C_ratio_omega': C_ratio_omega if 'C_ratio_omega' in dir() else np.nan,  # C/Cl for omega
        'L_real': L_real,
        'L_rand': L_rand,
        'L_ratio': L_ratio if 'L_ratio' in dir() else np.nan,  # Lr/L for omega
        'omega': omega,
        'sigma': sigma,
        'lcc_size': lcc_size,
        'lcc_coverage': lcc_coverage,
        'interpretation': interpretation,
        'n_random_graphs': n_random_graphs
    }

def small_worldness_analysis(graphs, years=[2014, 2024], n_random_graphs=10):
    """
    Perform small-worldness analysis for specified years using L06 methodology.
    
    Parameters:
    -----------
    graphs : dict
        Dictionary of NetworkX DiGraph objects keyed by "G_{year}"
    years : list
        Years to analyze
    n_random_graphs : int
        Number of random graphs for averaging
    
    Returns:
    --------
    dict
        Results for each year with small-world metrics
    """
    results = {}
    
    for year in years:
        G_key = f"G_{year}"
        if G_key not in graphs:
            print(f"Warning: Graph for year {year} not found, skipping...")
            continue
        
        G = graphs[G_key]
        print(f"  Computing small-worldness (L06) for {year}... (this may take a moment)")
        sw_metrics = calculate_small_worldness_l06(G, n_random_graphs=n_random_graphs)
        results[year] = sw_metrics
    
    return results

def plot_small_worldness_comparison(results, save_path=None):
    """
    Create a visual comparison of small-world metrics.
    
    Parameters:
    -----------
    results : dict
        Results from small_worldness_analysis
    save_path : str, optional
        Path to save the figure
    """
    years = sorted(results.keys())
    
    # Increase height to accommodate bottom legends
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # 1. Clustering coefficient comparison
    ax1 = axes[0]
    x = np.arange(len(years))
    width = 0.35
    c_real = [results[y]['C_real'] for y in years]
    c_rand = [results[y]['C_rand'] for y in years]
    
    bars1 = ax1.bar(x - width/2, c_real, width, label='Real Network', color='steelblue')
    bars2 = ax1.bar(x + width/2, c_rand, width, label='Random Graph', color='lightcoral')
    
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Clustering Coefficient (C)', fontsize=11)
    ax1.set_title('Clustering: Real vs Random', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    # Move legend below
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add ratio annotations
    for i, year in enumerate(years):
        ratio = results[year]['C_ratio']
        # Handle cases where bars might be very small
        height = max(c_real[i], c_rand[i])
        ax1.annotate(f'{ratio:.1f}x', xy=(i, height),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
    
    # 2. Path length comparison
    ax2 = axes[1]
    l_real = [results[y]['L_real'] for y in years]
    l_rand = [results[y]['L_rand'] for y in years]
    
    bars3 = ax2.bar(x - width/2, l_real, width, label='Real Network', color='steelblue')
    bars4 = ax2.bar(x + width/2, l_rand, width, label='Random Graph', color='lightcoral')
    
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Average Path Length (L)', fontsize=11)
    ax2.set_title('Path Length: Real vs Random', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    # Move legend below
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add ratio annotations
    for i, year in enumerate(years):
        ratio = results[year]['L_ratio']
        height = max(l_real[i], l_rand[i])
        ax2.annotate(f'{ratio:.2f}x', xy=(i, height),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
    
    # 3. Small-Worldness coefficients
    ax3 = axes[2]
    omega_vals = [results[y]['omega'] for y in years]
    sigma_vals = [results[y]['sigma'] for y in years]
    
    ax3.bar(x - width/2, omega_vals, width, label='ω (omega)', color='darkgreen')
    ax3.bar(x + width/2, sigma_vals, width, label='σ (sigma)', color='orange')
    
    # Add reference lines
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=1, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='σ=1 threshold')
    ax3.axhspan(-0.5, 0.5, alpha=0.1, color='green', label='Small-World zone')
    
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('Coefficient Value', fontsize=11)
    ax3.set_title('Small-Worldness Coefficients', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(years)
    # Move legend below
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Small-World Network Analysis: European Air Traffic', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust layout to make room for bottom legends
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig


# -----------------------------------------------------------------------------
# 5. CORE-PERIPHERY PROFILING (RICH-CLUB & K-CORE)
# -----------------------------------------------------------------------------
# Purpose: Identify the hierarchical structure of the network by analyzing
# the 'Core' (traditional hub airports) vs 'Periphery' (low-cost carrier airports).
#
# Methods:
# - K-Core Decomposition: Find the innermost core of highly connected nodes
# - Rich-Club Coefficient: Test if high-degree hubs preferentially connect to each other
# - Core vs Periphery Density: Compare edge density within/between layers

def core_periphery_analysis(G, weight=None):
    """
    Perform Core-Periphery analysis using K-Core decomposition.
    
    Parameters:
    -----------
    G : nx.DiGraph or nx.Graph
        NetworkX graph to analyze
    weight : str, optional
        Edge weight attribute (not used for k-core but useful for density)
    
    Returns:
    --------
    dict
        Dictionary with core-periphery metrics and node classifications
    """
    # Convert to undirected for k-core analysis
    G_undirected = G.to_undirected() if G.is_directed() else G.copy()
    
    # Remove self-loops (required for k-core analysis)
    G_undirected.remove_edges_from(nx.selfloop_edges(G_undirected))
    
    # K-Core decomposition
    core_numbers = nx.core_number(G_undirected)
    max_k = max(core_numbers.values()) if core_numbers else 0
    
    # Identify nodes in the maximum k-core (Inner Core)
    core_nodes = [node for node, k in core_numbers.items() if k == max_k]
    periphery_nodes = [node for node, k in core_numbers.items() if k < max_k]
    
    # Create subgraphs
    G_core = G_undirected.subgraph(core_nodes).copy() if core_nodes else nx.Graph()
    G_periphery = G_undirected.subgraph(periphery_nodes).copy() if periphery_nodes else nx.Graph()
    
    # Calculate densities
    n_core = len(core_nodes)
    n_periphery = len(periphery_nodes)
    n_total = G_undirected.number_of_nodes()
    
    # Core density: edges within the core
    core_density = nx.density(G_core) if n_core > 1 else 0
    
    # Periphery density: edges within the periphery
    periphery_density = nx.density(G_periphery) if n_periphery > 1 else 0
    
    # Core-Periphery edges: edges between core and periphery
    core_periphery_edges = 0
    for u, v in G_undirected.edges():
        if (u in core_nodes and v in periphery_nodes) or (v in core_nodes and u in periphery_nodes):
            core_periphery_edges += 1
    
    # Theoretical max core-periphery edges
    max_cp_edges = n_core * n_periphery if n_core > 0 and n_periphery > 0 else 1
    core_periphery_density = core_periphery_edges / max_cp_edges if max_cp_edges > 0 else 0
    
    # Calculate Rich-Club Coefficient
    try:
        rich_club = nx.rich_club_coefficient(G_undirected, normalized=False)
        # Get coefficients for key degree thresholds
        degrees = [d for n, d in G_undirected.degree()]
        avg_degree = np.mean(degrees)
        median_degree = np.median(degrees)
        
        # Find rich-club at different thresholds
        degree_thresholds = [int(np.percentile(degrees, p)) for p in [50, 75, 90]]
        rc_values = {k: rich_club.get(k, np.nan) for k in degree_thresholds if k in rich_club}
    except Exception as e:
        rich_club = {}
        rc_values = {}
        avg_degree = np.nan
        median_degree = np.nan
    
    # K-Core distribution
    k_core_distribution = {}
    for node, k in core_numbers.items():
        k_core_distribution[k] = k_core_distribution.get(k, 0) + 1
    
    return {
        'max_k_core': max_k,
        'n_core_nodes': n_core,
        'n_periphery_nodes': n_periphery,
        'core_fraction': n_core / n_total if n_total > 0 else 0,
        'core_nodes': core_nodes,
        'periphery_nodes': periphery_nodes,
        'core_density': core_density,
        'periphery_density': periphery_density,
        'core_periphery_density': core_periphery_density,
        'core_periphery_edges': core_periphery_edges,
        'rich_club_coefficients': rich_club,
        'rich_club_at_thresholds': rc_values,
        'k_core_distribution': k_core_distribution,
        'avg_degree': avg_degree,
        'median_degree': median_degree
    }


def core_periphery_profiling(graphs, years=[2014, 2024]):
    """
    Perform Core-Periphery profiling for specified years.
    
    Parameters:
    -----------
    graphs : dict
        Dictionary of NetworkX DiGraph objects keyed by "G_{year}"
    years : list
        Years to analyze
    
    Returns:
    --------
    dict
        Results for each year with core-periphery metrics
    """
    results = {}
    
    for year in years:
        G_key = f"G_{year}"
        if G_key not in graphs:
            print(f"Warning: Graph for year {year} not found, skipping...")
            continue
        
        G = graphs[G_key]
        print(f"  Computing Core-Periphery profile for {year}...")
        cp_metrics = core_periphery_analysis(G)
        results[year] = cp_metrics
    
    return results


def plot_core_periphery(G, cp_results, year, save_path=None):
    """
    Visualize the Core vs Periphery structure of the network.
    
    Parameters:
    -----------
    G : nx.DiGraph or nx.Graph
        NetworkX graph to visualize
    cp_results : dict
        Results from core_periphery_analysis
    year : int
        Year label for the plot
    save_path : str, optional
        Path to save the figure
    """
    G_undirected = G.to_undirected() if G.is_directed() else G.copy()
    
    core_nodes = set(cp_results['core_nodes'])
    periphery_nodes = set(cp_results['periphery_nodes'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. K-Core Distribution
    ax1 = axes[0]
    k_dist = cp_results['k_core_distribution']
    k_values = sorted(k_dist.keys())
    counts = [k_dist[k] for k in k_values]
    
    colors = ['steelblue' if k < cp_results['max_k_core'] else 'crimson' for k in k_values]
    ax1.bar(k_values, counts, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('K-Core Number', fontsize=11)
    ax1.set_ylabel('Number of Nodes', fontsize=11)
    ax1.set_title(f'K-Core Distribution ({year})\nMax K-Core: {cp_results["max_k_core"]}', 
                  fontsize=12, fontweight='bold')
    ax1.axvline(x=cp_results['max_k_core'], color='red', linestyle='--', 
                linewidth=2, label=f'Inner Core (k={cp_results["max_k_core"]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Density Comparison
    ax2 = axes[1]
    densities = [cp_results['core_density'], 
                 cp_results['periphery_density'], 
                 cp_results['core_periphery_density']]
    labels = ['Core\nDensity', 'Periphery\nDensity', 'Core-Periphery\nDensity']
    colors_bar = ['crimson', 'steelblue', 'mediumpurple']
    
    bars = ax2.bar(labels, densities, color=colors_bar, edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Edge Density', fontsize=11)
    ax2.set_title(f'Core vs Periphery Density ({year})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, densities):
        ax2.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
    
    # 3. Rich-Club Coefficient
    ax3 = axes[2]
    rc = cp_results['rich_club_coefficients']
    if rc:
        rc_keys = sorted([k for k in rc.keys() if k > 0])
        rc_vals = [rc[k] for k in rc_keys]
        
        ax3.plot(rc_keys, rc_vals, 'o-', color='darkgreen', linewidth=2, markersize=4)
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Neutral (φ=1)')
        ax3.set_xlabel('Degree Threshold (k)', fontsize=11)
        ax3.set_ylabel('Rich-Club Coefficient φ(k)', fontsize=11)
        ax3.set_title(f'Rich-Club Coefficient ({year})\nDo hubs connect to hubs?', 
                      fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Highlight zone where φ > 1 (rich-club effect)
        ax3.axhspan(1, max(rc_vals) * 1.1 if rc_vals else 2, alpha=0.1, color='green', 
                    label='Rich-Club Effect')
    else:
        ax3.text(0.5, 0.5, 'Rich-Club data\nnot available', 
                 ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title(f'Rich-Club Coefficient ({year})', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Core-Periphery Analysis: European Air Traffic Network ({year})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig


# =============================================================================
# MAIN EXECUTION: RUN ALL ANALYSES
# =============================================================================

def run_all_structural_analyses(graphs):
    """
    Execute all three structural analyses and return organized results.
    
    Parameters:
    -----------
    graphs : dict
        Dictionary of NetworkX DiGraph objects keyed by "G_{year}"
    
    Returns:
    --------
    dict
        All results organized for easy tabulation
    """
    print("=" * 70)
    print("STRUCTURAL ANALYSIS FOR AIR TRAFFIC NETWORK")
    print("=" * 70)
    
    # 1. BREXIT SENSITIVITY ANALYSIS
    print("\n" + "-" * 70)
    print("1. STRUCTURAL SENSITIVITY ANALYSIS (BREXIT CHECK)")
    print("-" * 70)
    
    brexit_results = structural_sensitivity_analysis(graphs, base_year=2014)
    
    print(f"\nYear Analyzed: {brexit_results['year']}")
    print(f"UK/GB Nodes Removed: {brexit_results['num_removed_nodes']}")
    print(f"Removed Nodes: {', '.join(brexit_results['removed_nodes'][:10])}{'...' if len(brexit_results['removed_nodes']) > 10 else ''}")
    print("\nOriginal Graph Metrics:")
    print(f"  - Nodes: {brexit_results['original']['nodes']}")
    print(f"  - Edges: {brexit_results['original']['edges']}")
    print(f"  - Density: {brexit_results['original']['density']:.6f}")
    print(f"  - Average Clustering: {brexit_results['original']['avg_clustering']:.6f}")
    print(f"  - Largest SCC Size: {brexit_results['original']['largest_scc_size']}")
    print("\nGhost Graph Metrics (UK/GB Removed):")
    print(f"  - Nodes: {brexit_results['ghost']['nodes']}")
    print(f"  - Edges: {brexit_results['ghost']['edges']}")
    print(f"  - Density: {brexit_results['ghost']['density']:.6f}")
    print(f"  - Average Clustering: {brexit_results['ghost']['avg_clustering']:.6f}")
    print(f"  - Largest SCC Size: {brexit_results['ghost']['largest_scc_size']}")
    print("\nPercentage Change (Original → Ghost):")
    print(f"  - Nodes: {brexit_results['pct_change']['nodes']:.2f}%")
    print(f"  - Edges: {brexit_results['pct_change']['edges']:.2f}%")
    print(f"  - Density: {brexit_results['pct_change']['density']:.2f}%")
    print(f"  - Average Clustering: {brexit_results['pct_change']['avg_clustering']:.2f}%")
    print(f"  - Largest SCC Size: {brexit_results['pct_change']['largest_scc_size']:.2f}%")
    
    # 2. DEGREE ASSORTATIVITY TIME-SERIES
    print("\n" + "-" * 70)
    print("2. DEGREE ASSORTATIVITY TIME-SERIES (HUB-AND-SPOKE ANALYSIS)")
    print("-" * 70)
    
    assortativity_df = calculate_assortativity_timeseries(graphs)
    
    print("\nAssortativity Coefficients by Year:")
    print("(Negative = Hub-and-Spoke; Near Zero = Point-to-Point; Positive = Assortative)")
    print("\n" + assortativity_df.to_string(index=False))
    
    # Calculate trend
    if len(assortativity_df) > 2:
        z = np.polyfit(assortativity_df['year'], assortativity_df['assortativity'], 1)
        trend = "INCREASING (toward Point-to-Point)" if z[0] > 0 else "DECREASING (toward stronger Hub-and-Spoke)"
        print(f"\nTrend Analysis: {trend}")
        print(f"Linear Slope: {z[0]:.6f} per year")
    
    # Plot assortativity
    plot_assortativity_timeseries(assortativity_df, save_path='assortativity_timeseries.png')
    
    # 3. WEIGHTED VS. BINARY CORRELATION
    print("\n" + "-" * 70)
    print("3. WEIGHTED VS. BINARY CORRELATION ANALYSIS")
    print("-" * 70)
    
    correlation_results = weighted_binary_correlation_analysis(graphs, years=[2014, 2024])
    
    print("\nCorrelation Analysis (Degree vs. Strength):")
    for year, data in sorted(correlation_results.items()):
        r = data['correlation']['total']['r']
        p = data['correlation']['total']['p']
        print(f"\n{year}:")
        print(f"  - Pearson r (Total): {r:.4f}")
        print(f"  - p-value: {p:.2e}")
        print(f"  - Nodes: {data['nodes']}, Edges: {data['edges']}")
    
    # Compare correlation strength
    if len(correlation_results) >= 2:
        r_2014 = correlation_results[2014]['correlation']['total']['r']
        r_2024 = correlation_results[2024]['correlation']['total']['r']
        change = r_2024 - r_2014
        interpretation = "WEAKENING" if change < 0 else "STRENGTHENING"
        print(f"\nCorrelation Change (2014 → 2024): {change:.4f}")
        print(f"Interpretation: Link between routes and passengers is {interpretation}")
        
        if change < 0:
            print("  → New routes at small airports do NOT necessarily shift systemic power")
        else:
            print("  → Network connectivity remains aligned with passenger flows")
    
    # Plot correlation
    plot_degree_strength_correlation(correlation_results, save_path='degree_strength_correlation.png')
    
    # 4. SMALL-WORLDNESS COEFFICIENT (ω) ANALYSIS - L06 METHODOLOGY
    print("\n" + "-" * 70)
    print("4. FORMAL SMALL-WORLDNESS COEFFICIENT (ω) ANALYSIS - L06")
    print("-" * 70)
    print("\nProving if European airspace is a Small-World network...")
    print("L06 Formula: ω = L/L_rand - C_reg/C")
    print("Interpretation: ω ≈ 0 → Small-World | ω > 0 → Random-like | ω < 0 → Lattice-like")
    
    small_world_results = small_worldness_analysis(graphs, years=[2014, 2024], n_random_graphs=10)
    
    print("\nSmall-Worldness Analysis Results:")
    for year, data in sorted(small_world_results.items()):
        print(f"\n{year}:")
        print(f"  Network Size: n={data['n']}, m={data['m']}, avg_degree={data['avg_degree']:.2f}")
        print(f"  Largest Connected Component: {data['lcc_size']} nodes ({data['lcc_coverage']:.1f}% coverage)")
        print(f"  Clustering Coefficient:")
        print(f"    - Real (C): {data['C_real']:.6f}")
        print(f"    - Random (C_rand): {data['C_rand']:.6f}")
        print(f"    - Lattice baseline (C_reg): {data['C_reg']:.6f}")
        print(f"    - Ratio (C/C_rand): {data['C_ratio']:.2f}x")
        print(f"  Average Path Length:")
        print(f"    - Real (L): {data['L_real']:.4f}")
        print(f"    - Random (L_rand): {data['L_rand']:.4f}")
        print(f"    - Ratio (L/L_rand): {data['L_ratio']:.4f}")
        print(f"  Small-Worldness Coefficients:")
        print(f"    - ω (omega, L06): {data['omega']:.4f}")
        print(f"    - σ (sigma, Humphries): {data['sigma']:.4f}")
        print(f"  → Interpretation: {data['interpretation']}")
    
    # Plot small-worldness comparison
    plot_small_worldness_comparison(small_world_results, save_path='small_worldness_analysis.png')
    
    # 5. CORE-PERIPHERY PROFILING (RICH-CLUB & K-CORE)
    print("\n" + "-" * 70)
    print("5. CORE-PERIPHERY PROFILING (RICH-CLUB & K-CORE)")
    print("-" * 70)
    print("\nIdentifying network hierarchy: Core hubs vs Periphery airports...")
    
    cp_results = core_periphery_profiling(graphs, years=[2014, 2024])
    
    print("\nCore-Periphery Analysis Results:")
    for year, data in sorted(cp_results.items()):
        print(f"\n{year}:")
        print(f"  K-Core Structure:")
        print(f"    - Maximum K-Core: {data['max_k_core']}")
        print(f"    - Core Nodes (k={data['max_k_core']}): {data['n_core_nodes']} ({data['core_fraction']*100:.1f}%)")
        print(f"    - Periphery Nodes: {data['n_periphery_nodes']}")
        print(f"  Edge Densities:")
        print(f"    - Core Density: {data['core_density']:.6f}")
        print(f"    - Periphery Density: {data['periphery_density']:.6f}")
        print(f"    - Core-Periphery Density: {data['core_periphery_density']:.6f}")
        print(f"  Rich-Club Analysis:")
        if data['rich_club_at_thresholds']:
            for k, rc in data['rich_club_at_thresholds'].items():
                print(f"    - φ(k={k}): {rc:.4f}")
        else:
            print(f"    - Rich-club coefficients computed (see plot)")
    
    # Plot Core-Periphery for 2024
    if 2024 in cp_results:
        plot_core_periphery(graphs['G_2024'], cp_results[2024], 2024, 
                           save_path='core_periphery_2024.png')
    
    # ==========================================================================
    # CREATE LATEX-READY SUMMARY TABLE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("LATEX-READY SUMMARY TABLE")
    print("=" * 70)
    
    # Brexit sensitivity table
    brexit_table = pd.DataFrame({
        'Metric': ['Network Density', 'Average Clustering', 'Largest SCC Size'],
        'Original (2014)': [
            f"{brexit_results['original']['density']:.6f}",
            f"{brexit_results['original']['avg_clustering']:.6f}",
            f"{brexit_results['original']['largest_scc_size']}"
        ],
        'Ghost (No UK/GB)': [
            f"{brexit_results['ghost']['density']:.6f}",
            f"{brexit_results['ghost']['avg_clustering']:.6f}",
            f"{brexit_results['ghost']['largest_scc_size']}"
        ],
        '% Change': [
            f"{brexit_results['pct_change']['density']:.2f}%",
            f"{brexit_results['pct_change']['avg_clustering']:.2f}%",
            f"{brexit_results['pct_change']['largest_scc_size']:.2f}%"
        ]
    })
    
    print("\nTable 1: Structural Sensitivity Analysis (Brexit Check)")
    print(brexit_table.to_latex(index=False))
    
    # Assortativity table
    print("\nTable 2: Degree Assortativity Time-Series")
    print(assortativity_df[['year', 'assortativity', 'nodes', 'edges']].to_latex(index=False))
    
    # Correlation comparison table
    corr_table = pd.DataFrame({
        'Year': [2014, 2024],
        'Pearson r': [
            correlation_results[2014]['correlation']['total']['r'],
            correlation_results[2024]['correlation']['total']['r']
        ],
        'p-value': [
            correlation_results[2014]['correlation']['total']['p'],
            correlation_results[2024]['correlation']['total']['p']
        ],
        'Nodes': [correlation_results[2014]['nodes'], correlation_results[2024]['nodes']],
        'Edges': [correlation_results[2014]['edges'], correlation_results[2024]['edges']]
    })
    
    print("\nTable 3: Weighted vs. Binary Correlation (Degree-Strength)")
    print(corr_table.to_latex(index=False))
    
    # Small-Worldness table (updated for L06)
    sw_table = pd.DataFrame({
        'Year': list(small_world_results.keys()),
        'C (Real)': [small_world_results[y]['C_real'] for y in small_world_results.keys()],
        'C (Rand)': [small_world_results[y]['C_rand'] for y in small_world_results.keys()],
        'C_reg': [small_world_results[y]['C_reg'] for y in small_world_results.keys()],
        'L (Real)': [small_world_results[y]['L_real'] for y in small_world_results.keys()],
        'L (Rand)': [small_world_results[y]['L_rand'] for y in small_world_results.keys()],
        'ω (L06)': [small_world_results[y]['omega'] for y in small_world_results.keys()],
        'σ': [small_world_results[y]['sigma'] for y in small_world_results.keys()],
        'Classification': [small_world_results[y]['interpretation'] for y in small_world_results.keys()]
    })
    
    print("\nTable 4: Small-Worldness Coefficient Analysis (L06 Methodology)")
    print(sw_table.to_latex(index=False))
    
    # Core-Periphery table
    cp_table = pd.DataFrame({
        'Year': list(cp_results.keys()),
        'Max K-Core': [cp_results[y]['max_k_core'] for y in cp_results.keys()],
        'Core Nodes': [cp_results[y]['n_core_nodes'] for y in cp_results.keys()],
        'Core %': [f"{cp_results[y]['core_fraction']*100:.1f}%" for y in cp_results.keys()],
        'Core Density': [cp_results[y]['core_density'] for y in cp_results.keys()],
        'Periphery Density': [cp_results[y]['periphery_density'] for y in cp_results.keys()],
        'CP Density': [cp_results[y]['core_periphery_density'] for y in cp_results.keys()]
    })
    
    print("\nTable 5: Core-Periphery Profiling (K-Core Analysis)")
    print(cp_table.to_latex(index=False))
    
    return {
        'brexit_sensitivity': brexit_results,
        'assortativity_timeseries': assortativity_df,
        'degree_strength_correlation': correlation_results,
        'small_worldness': small_world_results,
        'core_periphery': cp_results,
        'latex_tables': {
            'brexit': brexit_table,
            'assortativity': assortativity_df,
            'correlation': corr_table,
            'small_worldness': sw_table,
            'core_periphery': cp_table
        }
    }

# =============================================================================
# LOAD GRAPHS AND EXECUTE ANALYSIS
# =============================================================================


def load_graphs(years=range(2014, 2025)):
    """
    Load graphs from pickle files.
    
    Parameters:
    -----------
    years : range or list
        Years to load (default: 2014-2024)
    
    Returns:
    --------
    dict
        Dictionary of NetworkX DiGraph objects keyed by "G_{year}"
    """
    graphs = {}
    
    for year in years:
        pickle_file = f"G_{year}.pickle"
        if os.path.exists(pickle_file):
            graphs[f"G_{year}"] = pickle.load(open(pickle_file, 'rb'))
            print(f"Loaded {pickle_file}: {graphs[f'G_{year}'].number_of_nodes()} nodes, {graphs[f'G_{year}'].number_of_edges()} edges")
        else:
            print(f"Warning: {pickle_file} not found, skipping year {year}")
    
    return graphs

if __name__ == "__main__":
    print("Loading graphs from pickle files...")
    graphs = load_graphs()
    
    if not graphs:
        print("ERROR: No graphs loaded! Make sure the pickle files exist in the current directory.")
        print("Expected files: G_2014.pickle, G_2015.pickle, ..., G_2024.pickle")
    else:
        print(f"\nLoaded {len(graphs)} graphs successfully.\n")
        results = run_all_structural_analyses(graphs)