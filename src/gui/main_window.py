import os
from network import NeuronalNetwork
from parameters import NeuronParameters
from .plot_widgets import PlotWidget
from .neuron_config import NeuronConfigWidget
from .network_config import NetworkConfigWidget
from PyQt5.QtWidgets import QDoubleSpinBox
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QMessageBox, QSplitter, QGroupBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib
matplotlib.use('Qt5Agg')


class SimulationThread(QThread):
    """Thread for running simulations without blocking the GUI."""

    finished = pyqtSignal(object)
    progress = pyqtSignal(int)

    def __init__(self, network, duration):
        super().__init__()
        self.network = network
        self.duration = duration

    def run(self):
        try:
            # Simulate in chunks to allow progress updates
            chunk_duration = max(self.network.dt * 100, self.duration / 100)
            remaining = self.duration
            simulated = 0.0

            while remaining > 0:
                chunk = min(chunk_duration, remaining)
                self.network.simulate(chunk)
                simulated += chunk
                remaining -= chunk

                progress = int((simulated / self.duration) * 100)
                self.progress.emit(min(100, progress))

            self.progress.emit(100)
            self.finished.emit(self.network)

        except Exception as e:
            self.finished.emit(None)
            raise e


class NeuralNetworkGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.network = None
        self.simulation_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Neural Network Simulation GUI")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # splitter for resizable window parts
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left: config
        left_panel = self.create_config_panel()
        splitter.addWidget(left_panel)

        # Right: Visualization
        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 1000])
        self.apply_styles()

    def create_config_panel(self):
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(10, 10, 10, 10)
        config_layout.setSpacing(10)

        # Title
        title = QLabel("Configuration")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 5px;")
        config_layout.addWidget(title)

        config_tabs = QTabWidget()
        config_tabs.setTabPosition(QTabWidget.North)

        # Network config
        self.network_config = NetworkConfigWidget()
        self.network_config.num_neurons_spinbox.valueChanged.connect(
            lambda v: self.neuron_config.update_table_size(v)
        )
        config_tabs.addTab(self.network_config, "Network")

        # Neuron config
        self.neuron_config = NeuronConfigWidget()
        config_tabs.addTab(self.neuron_config, "Neurons")

        config_layout.addWidget(config_tabs)

        # Simulation
        sim_group = QGroupBox("Simulation Controls")
        sim_layout = QVBoxLayout(sim_group)

        # Duration input
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Duration (ms):"))
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setRange(1.0, 10000.0)
        self.duration_spinbox.setValue(200.0)
        self.duration_spinbox.setSingleStep(10.0)
        duration_layout.addWidget(self.duration_spinbox)
        sim_layout.addLayout(duration_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        sim_layout.addWidget(self.progress_bar)

        # Buttons
        button_layout = QHBoxLayout()

        # TODO: extract styles into separate file
        # Allow (and add) multiple themes...
        # TODO: Research nice colour palletes
        self.run_button = QPushButton("Run Simulation")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #2d5a2d;
                color: #90ee90;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                border: 1px solid #4CAF50;
            }
            QPushButton:hover {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #2d2d2d;
            }
        """)
        self.run_button.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #5a2d2d;
                color: #ff9999;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                border: 1px solid #f44336;
            }
            QPushButton:hover {
                background-color: #f44336;
                color: white;
            }
        """)
        self.reset_button.clicked.connect(self.reset_simulation)
        button_layout.addWidget(self.reset_button)

        sim_layout.addLayout(button_layout)
        config_layout.addWidget(sim_group)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            "padding: 5px; background-color: #252525; border: 1px solid #3d3d3d; border-radius: 3px;")
        config_layout.addWidget(self.status_label)

        config_layout.addStretch()

        return config_widget

    def create_visualization_panel(self):
        """Create the visualization panel with plots."""
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title = QLabel("Visualization")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 5px;")
        plot_layout.addWidget(title)

        # Plot tabs
        self.plot_widget = PlotWidget()
        plot_layout.addWidget(self.plot_widget)

        return plot_widget

    # TODO: same as above: extract styling
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #2d2d2d;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #b0b0b0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: 1px solid #3d3d3d;
            }
            QTabBar::tab:hover {
                background-color: #3d3d3d;
                color: #ffffff;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                color: #4a9eff;
                font-weight: bold;
                border-bottom: 2px solid #4a9eff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #e0e0e0;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4a9eff;
            }
            QLabel {
                color: #e0e0e0;
            }
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 4px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover, QComboBox:hover {
                border: 1px solid #4a9eff;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {
                border: 2px solid #4a9eff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #b0b0b0;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: #e0e0e0;
                selection-background-color: #4a9eff;
                selection-color: #ffffff;
                border: 1px solid #3d3d3d;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #3d3d3d;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border: 1px solid #4a9eff;
            }
            QPushButton:pressed {
                background-color: #1d1d1d;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #666666;
                border: 1px solid #2d2d2d;
            }
            QTableWidget {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3d3d3d;
                gridline-color: #3d3d3d;
                selection-background-color: #4a9eff;
                selection-color: #ffffff;
            }
            QTableWidget::item {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QTableWidget::item:selected {
                background-color: #4a9eff;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #252525;
                color: #e0e0e0;
                padding: 4px;
                border: 1px solid #3d3d3d;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                text-align: center;
                background-color: #252525;
                color: #e0e0e0;
            }
            QProgressBar::chunk {
                background-color: #4a9eff;
                border-radius: 2px;
            }
            QCheckBox {
                color: #e0e0e0;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #4a9eff;
            }
            QCheckBox::indicator:checked {
                background-color: #4a9eff;
                border: 1px solid #4a9eff;
            }
            QCheckBox::indicator:checked::after {
                content: "✓";
                color: white;
                font-weight: bold;
            }
        """)

    def run_simulation(self):
        """Run the simulation with current parameters."""
        try:
            # Get parameters from config widgets
            network_params = self.network_config.get_parameters()
            neuron_params_list = self.neuron_config.get_parameters()
            duration = self.duration_spinbox.value()

            # Create network
            parent_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Convert dict to NeuronParameters objects
            if neuron_params_list and isinstance(neuron_params_list[0], dict):
                neuron_params_objects = []
                for params_dict in neuron_params_list:
                    params = NeuronParameters()
                    for key, value in params_dict.items():
                        if hasattr(params, key):
                            setattr(params, key, value)
                    neuron_params_objects.append(params)
                neuron_params_list = neuron_params_objects

            self.network = NeuronalNetwork(
                num_neurons=network_params['num_neurons'],
                neuron_params_list=neuron_params_list if neuron_params_list else None
            )

            # Apply network topology
            if network_params['topology'] == 'feedforward':
                layers = self.network.create_feedforward_network(
                    network_params['layer_sizes'],
                    clear_existing=True
                )
                self._network_layers = layers
            elif network_params['topology'] == 'random':
                self.network.create_random_network(
                    connection_probability=network_params['connection_probability'],
                    weight_range=network_params['weight_range'],
                    clear_existing=True
                )
            elif network_params['topology'] == 'custom':
                # Add custom connections
                if 'connections' in network_params:
                    for pre, post, weight, delay in network_params['connections']:
                        if 0 <= pre < self.network.num_neurons and 0 <= post < self.network.num_neurons:
                            self.network.connect_neurons(
                                pre, post, weight=weight, delay=delay)

            # Add external inputs if specified
            if network_params.get('external_inputs'):
                for ext_input in network_params['external_inputs']:
                    self.network.add_external_input(
                        neuron_id=ext_input['neuron_id'],
                        compartment_id=ext_input['compartment_id'],
                        t_start=ext_input['t_start'],
                        t_end=ext_input['t_end'],
                        amplitude=ext_input['amplitude']
                    )

            # Disable run button and show progress
            self.run_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.status_label.setText("Running simulation...")

            # Run simulation in thread
            self.simulation_thread = SimulationThread(self.network, duration)
            self.simulation_thread.progress.connect(self.progress_bar.setValue)
            self.simulation_thread.finished.connect(self.simulation_finished)
            self.simulation_thread.start()

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to run simulation:\n{str(e)}")
            self.run_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"Error: {str(e)}")

    def simulation_finished(self, network):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)

        if network is None:
            self.status_label.setText("Simulation failed")
            QMessageBox.critical(
                self, "Error", "Simulation failed to complete.")
            return

        self.network = network
        self.status_label.setText("Simulation complete")

        # Try to get layers from network if it was created with feedforward topology
        layers = None
        if hasattr(self, '_network_layers'):
            layers = self._network_layers

        # Update plots
        self.plot_widget.update_plots(network, layers)

    def reset_simulation(self):
        if self.network:
            self.network.reset_to_initial_state()
            self.plot_widget.clear_plots()
            self.status_label.setText("Simulation reset")
        else:
            self.status_label.setText("No simulation to reset")

    def closeEvent(self, event):
        if self.simulation_thread and self.simulation_thread.isRunning():
            reply = QMessageBox.question(
                self, "Simulation Running",
                "A simulation is currently running. Do you want to quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.simulation_thread.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = NeuralNetworkGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
