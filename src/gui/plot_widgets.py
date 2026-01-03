import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import networkx as nx
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget
import matplotlib
from collections import defaultdict
from matplotlib.gridspec import GridSpec
matplotlib.use('Qt5Agg')

# Use dark theme for plots
plt.style.use('dark_background')
sns.set_palette("Set2")


class PlotCanvas(FigureCanvas):
    """Base canvas for plots."""

    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#1e1e1e')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = None

    def clear(self):
        self.fig.clear()
        self.fig.patch.set_facecolor('#1e1e1e')
        self.draw()


class PlotWidget(QWidget):
    """Main widget containing all plot tabs."""

    def __init__(self):
        super().__init__()
        self.network = None
        self.layers = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        # Create individual plot canvases
        self.network_arch_canvas = PlotCanvas(self, width=8, height=6)
        self.raster_canvas = PlotCanvas(self, width=8, height=6)
        self.firing_rate_canvas = PlotCanvas(self, width=8, height=6)
        self.connection_matrix_canvas = PlotCanvas(self, width=8, height=6)
        self.parameter_comparison_canvas = PlotCanvas(self, width=8, height=6)
        self.network_stats_canvas = PlotCanvas(self, width=8, height=6)
        self.all_plots_canvas = PlotCanvas(self, width=16, height=10)

        # Add tabs
        self.tabs.addTab(self.network_arch_canvas, "Network Architecture")
        self.tabs.addTab(self.raster_canvas, "Raster Plot")
        self.tabs.addTab(self.firing_rate_canvas, "Firing Rates")
        self.tabs.addTab(self.connection_matrix_canvas, "Connection Matrix")
        self.tabs.addTab(self.parameter_comparison_canvas,
                         "Parameter Comparison")
        self.tabs.addTab(self.network_stats_canvas, "Network Statistics")
        self.tabs.addTab(self.all_plots_canvas, "All Plots")

        layout.addWidget(self.tabs)

    def update_plots(self, network, layers=None):
        self.network = network
        self.layers = layers if layers is not None else self._infer_layers(
            network)

        # Update each plots
        self.plot_network_architecture()
        self.plot_raster()
        self.plot_firing_rates()
        self.plot_connection_matrix()
        self.plot_parameter_comparison()
        self.plot_network_statistics()
        self.plot_all()

    def clear_plots(self):
        for i in range(self.tabs.count()):
            canvas = self.tabs.widget(i)
            if isinstance(canvas, PlotCanvas):
                canvas.clear()

    def _infer_layers(self, network):
        # TODO: make better
        num_neurons = network.num_neurons
        if num_neurons == 8:
            return [[0, 1], [2, 3, 4], [5, 6], [7]]
        else:
            # Default: all in one layer
            return [list(range(num_neurons))]

    def _apply_dark_theme(self, ax):
        # TODO: as stated elsewhere: extract color themes...
        ax.set_facecolor('#2d2d2d')
        ax.spines['bottom'].set_color('#666666')
        ax.spines['top'].set_color('#666666')
        ax.spines['right'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        ax.tick_params(colors='#b0b0b0')
        ax.xaxis.label.set_color('#e0e0e0')
        ax.yaxis.label.set_color('#e0e0e0')
        ax.title.set_color('#e0e0e0')
        ax.grid(True, alpha=0.2, color='#666666')

    def plot_network_architecture(self):
        """Use networkx for plotting the network architecture"""
        if self.network is None:
            return

        self.network_arch_canvas.clear()
        ax = self.network_arch_canvas.fig.add_subplot(111)
        self._apply_dark_theme(ax)

        G = nx.DiGraph()

        for i in range(len(self.network.neurons)):
            G.add_node(i)

        for i in range(len(self.network.neurons)):
            for j in range(len(self.network.neurons)):
                if self.network.connection_matrix[i, j] != 0:
                    G.add_edge(
                        i, j, weight=self.network.connection_matrix[i, j])

        # Create layout
        if self.layers and len(self.layers) > 1:
            pos = {}
            for layer_idx, nodes in enumerate(self.layers):
                n = len(nodes)
                y_coords = [((n - 1) / 2) - i for i in range(n)
                            ] if n > 1 else [0]

                for node, y in zip(nodes, y_coords):
                    pos[node] = (layer_idx, y)
        else:
            pos = nx.spring_layout(G)

        # Color nodes by layer (dark theme colors)
        node_colors = []
        for i in range(len(self.network.neurons)):
            color = '#666666'
            if self.layers:
                for layer_idx, layer_nodes in enumerate(self.layers):
                    if i in layer_nodes:
                        colors = ['#4a9eff', '#4CAF50', '#f44336',
                                  '#ffa726', '#ab47bc', '#26c6da']
                        color = colors[layer_idx % len(colors)]
                        break
            node_colors.append(color)

        nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
                node_size=900, arrows=True, arrowstyle='->', arrowsize=20,
                edge_color='#666666', width=2, alpha=0.7, font_color='#e0e0e0')

        edge_labels = nx.get_edge_attributes(G, 'weight')
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels, ax=ax, font_size=8, font_color='#e0e0e0')

        ax.set_title('Network Architecture', fontweight='bold', fontsize=14)
        ax.axis('off')

        self.network_arch_canvas.draw()

    def plot_raster(self):
        if self.network is None:
            return

        self.raster_canvas.clear()
        ax = self.raster_canvas.fig.add_subplot(111)
        self._apply_dark_theme(ax)

        colors = ['#4a9eff', '#4a9eff', '#4CAF50', '#4CAF50',
                  '#4CAF50', '#f44336', '#f44336', '#ffa726']

        for i, neuron in enumerate(self.network.neurons):
            soma_id = neuron.soma_id
            if soma_id is not None:
                spike_times = neuron.compartments[soma_id].get_spike_times()
                color = colors[i % len(colors)]
                for spike_time in spike_times:
                    ax.scatter(spike_time, i, color=color, s=30, alpha=0.8)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron ID')
        ax.set_title('Raster Plot', fontweight='bold')
        ax.set_yticks(range(len(self.network.neurons)))
        ax.set_yticklabels([f'N{i}' for i in range(len(self.network.neurons))])
        ax.grid(True, alpha=0.3)

        self.raster_canvas.draw()

    def plot_firing_rates(self):
        if self.network is None:
            return

        self.firing_rate_canvas.clear()
        ax = self.firing_rate_canvas.fig.add_subplot(111)
        self._apply_dark_theme(ax)

        firing_rates = []
        for neuron in self.network.neurons:
            soma_id = neuron.soma_id
            if soma_id is not None:
                spike_times = neuron.compartments[soma_id].get_spike_times()
                firing_rate = len(spike_times) / (self.network.t /
                                                  1000.0) if self.network.t > 0 else 0
                firing_rates.append(firing_rate)
            else:
                firing_rates.append(0)

        colors = ['blue', 'blue', 'green', 'green',
                  'green', 'red', 'red', 'orange']
        bars = ax.bar(range(len(firing_rates)), firing_rates,
                      color=[colors[i % len(colors)]
                             for i in range(len(firing_rates))],
                      alpha=0.7, edgecolor='black')

        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title('Firing Rate Analysis', fontweight='bold')
        ax.set_xticks(range(len(firing_rates)))
        ax.set_xticklabels([f'N{i}' for i in range(len(firing_rates))])
        ax.grid(True, alpha=0.3, axis='y')

        for i, (bar, rate) in enumerate(zip(bars, firing_rates)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{rate:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#e0e0e0')

        self.firing_rate_canvas.draw()

    def plot_connection_matrix(self):
        if self.network is None:
            return

        self.connection_matrix_canvas.clear()
        ax = self.connection_matrix_canvas.fig.add_subplot(111)
        self._apply_dark_theme(ax)

        im = ax.imshow(self.network.connection_matrix,
                       cmap='coolwarm', aspect='auto')
        ax.set_xlabel('Post-synaptic')
        ax.set_ylabel('Pre-synaptic')
        ax.set_title('Connection Matrix\n(Weights)', fontweight='bold')
        ax.set_xticks(range(len(self.network.neurons)))
        ax.set_yticks(range(len(self.network.neurons)))
        ax.set_xticklabels([f'N{i}' for i in range(len(self.network.neurons))])
        ax.set_yticklabels([f'N{i}' for i in range(len(self.network.neurons))])

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Weight', fontsize=8)

        self.connection_matrix_canvas.draw()

    def plot_parameter_comparison(self):
        if self.network is None:
            return

        self.parameter_comparison_canvas.clear()
        ax = self.parameter_comparison_canvas.fig.add_subplot(111)
        self._apply_dark_theme(ax)

        thresholds = [
            neuron.params.spike_threshold for neuron in self.network.neurons]
        refractory_periods = [
            neuron.params.refractory_period for neuron in self.network.neurons]

        ax.scatter(thresholds, refractory_periods, c=range(len(thresholds)),
                   cmap='plasma', s=100, alpha=0.7)
        ax.set_xlabel('Spike Threshold (mV)')
        ax.set_ylabel('Refractory Period (ms)')
        ax.set_title('Neuron Parameter Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Annotate points
        param_groups = defaultdict(list)
        for i, (thresh, ref) in enumerate(zip(thresholds, refractory_periods)):
            param_groups[(thresh, ref)].append(i)

        for (thresh, ref), neuron_indices in param_groups.items():
            if len(neuron_indices) == 1:
                ax.annotate(f'N{neuron_indices[0]}', (thresh, ref),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            else:
                for j, neuron_idx in enumerate(neuron_indices):
                    angle = 2 * np.pi * j / len(neuron_indices)
                    offset_x = 8 * np.cos(angle)
                    offset_y = 8 * np.sin(angle)
                    ax.annotate(f'N{neuron_idx}', (thresh, ref),
                                xytext=(offset_x, offset_y),
                                textcoords='offset points', fontsize=8,
                                ha='center', va='center')

        self.parameter_comparison_canvas.draw()

    def plot_network_statistics(self):
        if self.network is None:
            return

        self.network_stats_canvas.clear()
        ax = self.network_stats_canvas.fig.add_subplot(111)
        self._apply_dark_theme(ax)

        firing_rates = []
        for neuron in self.network.neurons:
            soma_id = neuron.soma_id
            if soma_id is not None:
                spike_times = neuron.compartments[soma_id].get_spike_times()
                firing_rate = len(spike_times) / (self.network.t /
                                                  1000.0) if self.network.t > 0 else 0
                firing_rates.append(firing_rate)
            else:
                firing_rates.append(0)

        stats = {
            'Total Neurons': self.network.num_neurons,
            'Total Connections': len(self.network.connections),
            'Connection Density': f"{len(self.network.connections) / (self.network.num_neurons * max(1, self.network.num_neurons - 1)):.2f}",
            'Active Neurons': sum(1 for rate in firing_rates if rate > 0),
            'Avg Firing Rate': f"{np.mean(firing_rates):.1f} Hz",
            'Max Firing Rate': f"{np.max(firing_rates):.1f} Hz" if firing_rates else "0.0 Hz",
            'Simulation Time': f"{self.network.t:.1f} ms"
        }

        text = "Statistics:\n\n"
        for key, value in stats.items():
            text += f"• {key}: {value}\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', color='#e0e0e0',
                bbox=dict(boxstyle='round', facecolor='#252525', edgecolor='#3d3d3d', alpha=0.9))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Model Summary', fontweight='bold')

        self.network_stats_canvas.draw()

    def plot_all(self):
        if self.network is None:
            return

        self.all_plots_canvas.clear()
        fig = self.all_plots_canvas.fig

        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Network Architecture
        ax0 = fig.add_subplot(gs[0, 0])
        self._apply_dark_theme(ax0)
        self._plot_network_arch_ax(ax0)

        # Raster Plot
        ax1 = fig.add_subplot(gs[0, 1])
        self._apply_dark_theme(ax1)
        self._plot_raster_ax(ax1)

        # Firing Rate Analysis
        ax2 = fig.add_subplot(gs[0, 2])
        self._apply_dark_theme(ax2)
        self._plot_firing_rates_ax(ax2)

        # Connection Matrix
        ax3 = fig.add_subplot(gs[1, 0])
        self._apply_dark_theme(ax3)
        self._plot_connection_matrix_ax(ax3)

        # Parameter Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        self._apply_dark_theme(ax4)
        self._plot_parameter_comparison_ax(ax4)

        # Network Statistics
        ax5 = fig.add_subplot(gs[1, 2])
        self._apply_dark_theme(ax5)
        self._plot_network_statistics_ax(ax5)

        fig.suptitle('Multi-Compartment Neuronal Network Model',
                     fontsize=16, fontweight='bold', y=0.98, color='#e0e0e0')

        self.all_plots_canvas.draw()

    def _plot_network_arch_ax(self, ax):
        G = nx.DiGraph()
        for i in range(len(self.network.neurons)):
            G.add_node(i)
        for i in range(len(self.network.neurons)):
            for j in range(len(self.network.neurons)):
                if self.network.connection_matrix[i, j] != 0:
                    G.add_edge(
                        i, j, weight=self.network.connection_matrix[i, j])

        if self.layers and len(self.layers) > 1:
            pos = {}
            for layer_idx, nodes in enumerate(self.layers):
                n = len(nodes)
                y_coords = [((n - 1) / 2) - i for i in range(n)
                            ] if n > 1 else [0]
                for node, y in zip(nodes, y_coords):
                    pos[node] = (layer_idx, y)
        else:
            pos = nx.spring_layout(G)

        # Dark theme colors
        node_colors = []
        for i in range(len(self.network.neurons)):
            color = '#666666'
            if self.layers:
                for layer_idx, layer_nodes in enumerate(self.layers):
                    if i in layer_nodes:
                        colors = ['#4a9eff', '#4CAF50', '#f44336',
                                  '#ffa726', '#ab47bc', '#26c6da']
                        color = colors[layer_idx % len(colors)]
                        break
            node_colors.append(color)

        nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
                node_size=900, arrows=True, arrowstyle='->', arrowsize=20,
                edge_color='#666666', width=2, alpha=0.7, font_color='#e0e0e0')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels, ax=ax, font_size=8, font_color='#e0e0e0')
        ax.set_title('Network Architecture', fontweight='bold', fontsize=14)
        ax.axis('off')

    def _plot_raster_ax(self, ax):
        colors = ['#4a9eff', '#4a9eff', '#4CAF50', '#4CAF50',
                  '#4CAF50', '#f44336', '#f44336', '#ffa726']
        for i, neuron in enumerate(self.network.neurons):
            soma_id = neuron.soma_id
            if soma_id is not None:
                spike_times = neuron.compartments[soma_id].get_spike_times()
                color = colors[i % len(colors)]
                for spike_time in spike_times:
                    ax.scatter(spike_time, i, color=color, s=30, alpha=0.8)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Neuron ID')
        ax.set_title('Raster Plot', fontweight='bold')
        ax.set_yticks(range(len(self.network.neurons)))
        ax.set_yticklabels([f'N{i}' for i in range(len(self.network.neurons))])
        ax.grid(True, alpha=0.3)

    def _plot_firing_rates_ax(self, ax):
        firing_rates = []
        for neuron in self.network.neurons:
            soma_id = neuron.soma_id
            if soma_id is not None:
                spike_times = neuron.compartments[soma_id].get_spike_times()
                firing_rate = len(spike_times) / (self.network.t /
                                                  1000.0) if self.network.t > 0 else 0
                firing_rates.append(firing_rate)
            else:
                firing_rates.append(0)

        colors = ['#4a9eff', '#4a9eff', '#4CAF50', '#4CAF50',
                  '#4CAF50', '#f44336', '#f44336', '#ffa726']
        bars = ax.bar(range(len(firing_rates)), firing_rates,
                      color=[colors[i % len(colors)]
                             for i in range(len(firing_rates))],
                      alpha=0.7, edgecolor='#666666')
        ax.set_xlabel('Neuron ID')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title('Firing Rate Analysis', fontweight='bold')
        ax.set_xticks(range(len(firing_rates)))
        ax.set_xticklabels([f'N{i}' for i in range(len(firing_rates))])
        ax.grid(True, alpha=0.3, axis='y')
        for i, (bar, rate) in enumerate(zip(bars, firing_rates)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{rate:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#e0e0e0')

    def _plot_connection_matrix_ax(self, ax):
        im = ax.imshow(self.network.connection_matrix,
                       cmap='coolwarm', aspect='auto')
        ax.set_xlabel('Post-synaptic')
        ax.set_ylabel('Pre-synaptic')
        ax.set_title('Connection Matrix\n(Weights)', fontweight='bold')
        ax.set_xticks(range(len(self.network.neurons)))
        ax.set_yticks(range(len(self.network.neurons)))
        ax.set_xticklabels([f'N{i}' for i in range(len(self.network.neurons))])
        ax.set_yticklabels([f'N{i}' for i in range(len(self.network.neurons))])
        plt.colorbar(im, ax=ax, shrink=0.8).set_label('Weight', fontsize=8)

    def _plot_parameter_comparison_ax(self, ax):
        thresholds = [
            neuron.params.spike_threshold for neuron in self.network.neurons]
        refractory_periods = [
            neuron.params.refractory_period for neuron in self.network.neurons]
        ax.scatter(thresholds, refractory_periods, c=range(len(thresholds)),
                   cmap='plasma', s=100, alpha=0.7)
        ax.set_xlabel('Spike Threshold (mV)')
        ax.set_ylabel('Refractory Period (ms)')
        ax.set_title('Neuron Parameter Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
        param_groups = defaultdict(list)
        for i, (thresh, ref) in enumerate(zip(thresholds, refractory_periods)):
            param_groups[(thresh, ref)].append(i)
        for (thresh, ref), neuron_indices in param_groups.items():
            if len(neuron_indices) == 1:
                ax.annotate(f'N{neuron_indices[0]}', (thresh, ref),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            else:
                for j, neuron_idx in enumerate(neuron_indices):
                    angle = 2 * np.pi * j / len(neuron_indices)
                    offset_x = 8 * np.cos(angle)
                    offset_y = 8 * np.sin(angle)
                    ax.annotate(f'N{neuron_idx}', (thresh, ref),
                                xytext=(offset_x, offset_y),
                                textcoords='offset points', fontsize=8,
                                ha='center', va='center')

    def _plot_network_statistics_ax(self, ax):
        """Helper to plot network statistics on given axis."""
        firing_rates = []
        for neuron in self.network.neurons:
            soma_id = neuron.soma_id
            if soma_id is not None:
                spike_times = neuron.compartments[soma_id].get_spike_times()
                firing_rate = len(spike_times) / (self.network.t /
                                                  1000.0) if self.network.t > 0 else 0
                firing_rates.append(firing_rate)
            else:
                firing_rates.append(0)

        stats = {
            'Total Neurons': self.network.num_neurons,
            'Total Connections': len(self.network.connections),
            'Connection Density': f"{len(self.network.connections) / (self.network.num_neurons * max(1, self.network.num_neurons - 1)):.2f}",
            'Active Neurons': sum(1 for rate in firing_rates if rate > 0),
            'Avg Firing Rate': f"{np.mean(firing_rates):.1f} Hz",
            'Max Firing Rate': f"{np.max(firing_rates):.1f} Hz" if firing_rates else "0.0 Hz",
            'Simulation Time': f"{self.network.t:.1f} ms"
        }

        text = "Statistics:\n\n"
        for key, value in stats.items():
            text += f"• {key}: {value}\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace', color='#e0e0e0',
                bbox=dict(boxstyle='round', facecolor='#252525', edgecolor='#3d3d3d', alpha=0.9))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Model Summary', fontweight='bold')
