
import numpy as np
import seaborn as sns
import networkx as nx

from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from test_helper import create_network


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


LAYER_NAMES = ["Input", "Input", "Hidden1", "Hidden1",
               "Hidden1", "Hidden2", "Hidden2", "Output"]
COLORS = ['blue', 'blue', 'green', 'green', 'green', 'red', 'red', 'orange']


def run_demo():
    print("\nMulti-Compartment Neuronal Network Model")
    print("=" * 40)

    print("1. Creating network with multiple neuron types...")
    network, layers = create_network()

    soma_id_0 = network.neurons[0].soma_id
    soma_id_1 = network.neurons[1].soma_id
    network.add_external_input(
        neuron_id=0, compartment_id=soma_id_0, t_start=5.0, t_end=15.0, amplitude=0.05)
    network.add_external_input(
        neuron_id=1, compartment_id=soma_id_1, t_start=50.0, t_end=60.0, amplitude=0.01)

    simulation_duration = 200.0
    print(f"2. Run simulation for {simulation_duration} ms...")
    network.simulate(duration=simulation_duration)
    print("3. Simulation complete...")

    print("4. Create plots...")
    create_plots(network, layers)

    print_summary(network)


def create_plots(network, layers):
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)  # plot in 2x3 grid

    # Network Architecture
    ax0 = fig.add_subplot(gs[0, 0])
    plot_network_architecture(ax0, network, layers)

    # Raster Plot
    ax1 = fig.add_subplot(gs[0, 1])
    create_raster_plot(ax1, network)

    # Firing Rate Analysis
    ax2 = fig.add_subplot(gs[0, 2])
    plot_firing_rates(ax2, network)

    # Connection Matrix
    ax3 = fig.add_subplot(gs[1, 0])
    plot_connection_matrix(ax3, network)

    # Parameter Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    plot_parameter_comparison(ax4, network)

    # Network Statistics
    ax5 = fig.add_subplot(gs[1, 2])
    plot_network_statistics(ax5, network)

    plt.suptitle('Multi-Compartment Neuronal Network Model',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('../imgs/neural_network_demo.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_network_architecture(ax, network, layers):
    G = nx.DiGraph()

    for i in range(len(network.neurons)):
        G.add_node(i)

    for i in range(len(network.neurons)):
        for j in range(len(network.neurons)):
            if network.connection_matrix[i, j] != 0:
                G.add_edge(i, j, weight=network.connection_matrix[i, j])

    layer_assignment = {
        0: layers[0],  # Input layer
        1: layers[1],  # Hidden1 layer
        2: layers[2],  # Hidden2 layer
        3: layers[3]   # Output layer
    }

    pos = {}
    for layer, nodes in layer_assignment.items():
        n = len(nodes)
        y_coords = [((n - 1) / 2) - i for i in range(n)]  # center

        for node, y in zip(nodes, y_coords):
            pos[node] = (layer, y)

    node_colors = []
    for i in range(len(network.neurons)):
        if i in layers[0]:
            node_colors.append('lightblue')
        elif i in layers[1]:
            node_colors.append('lightgreen')
        elif i in layers[2]:
            node_colors.append('lightcoral')
        elif i in layers[3]:
            node_colors.append('gold')
        else:
            node_colors.append('lightgray')

    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors, node_size=900,
            arrows=True, arrowstyle='->', arrowsize=20, edge_color='gray', width=2, alpha=0.7)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)

    ax.set_title('Network Architecture', fontweight='bold', fontsize=14)
    ax.axis('off')


def create_raster_plot(ax, network):
    for i, neuron in enumerate(network.neurons):
        soma_id = neuron.soma_id
        spike_times = neuron.compartments[soma_id].get_spike_times()

        for spike_time in spike_times:
            ax.scatter(spike_time, i, color=COLORS[i], s=30, alpha=0.8)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Raster Plot', fontweight='bold')
    ax.set_yticks(range(8))
    ax.set_yticklabels([f'N{i}' for i in range(8)])
    ax.grid(True, alpha=0.3)


def plot_firing_rates(ax, network):
    firing_rates = []
    for neuron in network.neurons:
        soma_id = neuron.soma_id
        spike_times = neuron.compartments[soma_id].get_spike_times()
        firing_rate = len(spike_times) / (network.t /
                                          1000.0) if network.t > 0 else 0
        firing_rates.append(firing_rate)

    bars = ax.bar(range(8), firing_rates, color=COLORS,
                  alpha=0.7, edgecolor='black')
    ax.set_xlabel('Neuron ID')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Firing Rate Analysis', fontweight='bold')
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'N{i}' for i in range(8)])
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, rate) in enumerate(zip(bars, firing_rates)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{rate:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')


def plot_connection_matrix(ax, network):
    im = ax.imshow(network.connection_matrix, cmap='coolwarm', aspect='auto')
    ax.set_xlabel('Post-synaptic')
    ax.set_ylabel('Pre-synaptic')
    ax.set_title('Connection Matrix\n(Weights)', fontweight='bold')
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels([f'N{i}' for i in range(8)])
    ax.set_yticklabels([f'N{i}' for i in range(8)])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Weight', fontsize=8)


def plot_parameter_comparison(ax, network):
    thresholds = [neuron.params.spike_threshold for neuron in network.neurons]
    refractory_periods = [
        neuron.params.refractory_period for neuron in network.neurons]

    ax.scatter(thresholds, refractory_periods, c=range(
        8), cmap='plasma', s=100, alpha=0.7)
    ax.set_xlabel('Spike Threshold (mV)')
    ax.set_ylabel('Refractory Period (ms)')
    ax.set_title('Neuron Parameter Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3)

    param_groups = defaultdict(list)
    for i, (thresh, ref) in enumerate(zip(thresholds, refractory_periods)):
        param_groups[(thresh, ref)].append(i)

    for (thresh, ref), neuron_indices in param_groups.items():
        if len(neuron_indices) == 1:
            ax.annotate(f'N{neuron_indices[0]}', (thresh, ref), xytext=(
                5, 5), textcoords='offset points', fontsize=8)
        else:
            for j, neuron_idx in enumerate(neuron_indices):
                # if multiple neurons at the same position, spread lables
                angle = 2 * np.pi * j / len(neuron_indices)
                offset_x = 8 * np.cos(angle)
                offset_y = 8 * np.sin(angle)
                ax.annotate(f'N{neuron_idx}', (thresh, ref), xytext=(
                    offset_x, offset_y), textcoords='offset points', fontsize=8, ha='center', va='center')


def plot_network_statistics(ax, network):
    firing_rates = []
    for neuron in network.neurons:
        soma_id = neuron.soma_id
        spike_times = neuron.compartments[soma_id].get_spike_times()
        firing_rate = len(spike_times) / (network.t /
                                          1000.0) if network.t > 0 else 0
        firing_rates.append(firing_rate)

    stats = {
        'Total Neurons': network.num_neurons,
        'Total Connections': len(network.connections),
        'Connection Density': f"{len(network.connections) / (8 * 7):.2f}",
        'Active Neurons': sum(1 for rate in firing_rates if rate > 0),
        'Avg Firing Rate': f"{np.mean(firing_rates):.1f} Hz",
        'Max Firing Rate': f"{np.max(firing_rates):.1f} Hz",
        'Simulation Time': f"{network.t:.1f} ms"
    }

    text = "Statistics:\n\n"
    for key, value in stats.items():
        text += f"• {key}: {value}\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Model Summary', fontweight='bold')


def print_summary(network):
    print("\nModel Summary:")
    print("-" * 40)

    firing_rates = []
    for neuron in network.neurons:
        soma_id = neuron.soma_id
        spike_times = neuron.compartments[soma_id].get_spike_times()
        firing_rate = len(spike_times) / (network.t /
                                          1000.0) if network.t > 0 else 0
        firing_rates.append(firing_rate)

    print(f"- Multi-compartment neurons: {network.num_neurons} neurons")
    print(f"- Parameters: Different thresholds, refractory periods")
    print(f"- Synaptic connections: {len(network.connections)} connections")
    print(f"- Dynamics: Hodgkin-Huxley channels")
    print(
        f"- Network activity: {sum(1 for rate in firing_rates if rate > 0)} active neurons")
    print(f"- Firing rates: {np.min(firing_rates)          :.1f} - {np.max(firing_rates):.1f} Hz")
    print(f"- Simulation time: {network.t:.1f} ms\n")


if __name__ == "__main__":
    run_demo()
