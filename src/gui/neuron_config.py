from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt


class NeuronConfigWidget(QWidget):
    """Widget for configuring neuron parameters."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Parameter mode
        mode_group = QGroupBox("Parameter Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.uniform_mode = QCheckBox("Use uniform parameters for all neurons")
        self.uniform_mode.setChecked(True)
        self.uniform_mode.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.uniform_mode)

        layout.addWidget(mode_group)

        # Uniform parameters
        self.uniform_group = QGroupBox("Uniform Neuron Parameters")
        uniform_layout = QVBoxLayout(self.uniform_group)

        # Simulation parameters
        sim_params_group = QGroupBox("Simulation Parameters")
        sim_layout = QVBoxLayout(sim_params_group)

        dt_layout = QHBoxLayout()
        dt_layout.addWidget(QLabel("Time Step (ms):"))
        self.dt_spinbox = QDoubleSpinBox()
        self.dt_spinbox.setRange(0.001, 1.0)
        self.dt_spinbox.setSingleStep(0.01)
        self.dt_spinbox.setValue(0.01)
        self.dt_spinbox.setDecimals(3)
        dt_layout.addWidget(self.dt_spinbox)
        sim_layout.addLayout(dt_layout)

        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature (°C):"))
        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setRange(0.0, 50.0)
        self.temp_spinbox.setSingleStep(0.1)
        self.temp_spinbox.setValue(37.0)
        temp_layout.addWidget(self.temp_spinbox)
        sim_layout.addLayout(temp_layout)

        uniform_layout.addWidget(sim_params_group)

        # Spike parameters
        spike_params_group = QGroupBox("Spike Parameters")
        spike_layout = QVBoxLayout(spike_params_group)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Spike Threshold (mV):"))
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(-100.0, 100.0)
        self.threshold_spinbox.setSingleStep(1.0)
        self.threshold_spinbox.setValue(0.0)
        threshold_layout.addWidget(self.threshold_spinbox)
        spike_layout.addLayout(threshold_layout)

        refrac_layout = QHBoxLayout()
        refrac_layout.addWidget(QLabel("Refractory Period (ms):"))
        self.refrac_spinbox = QDoubleSpinBox()
        self.refrac_spinbox.setRange(0.0, 1000.0)
        self.refrac_spinbox.setSingleStep(1.0)
        self.refrac_spinbox.setValue(50.0)
        refrac_layout.addWidget(self.refrac_spinbox)
        spike_layout.addLayout(refrac_layout)

        uniform_layout.addWidget(spike_params_group)

        # Morphology parameters
        morph_params_group = QGroupBox("Morphology Parameters")
        morph_layout = QVBoxLayout(morph_params_group)

        dend_len_layout = QHBoxLayout()
        dend_len_layout.addWidget(QLabel("Dendrite Length (μm):"))
        self.dend_len_spinbox = QDoubleSpinBox()
        self.dend_len_spinbox.setRange(1.0, 10000.0)
        self.dend_len_spinbox.setSingleStep(10.0)
        self.dend_len_spinbox.setValue(200.0)
        dend_len_layout.addWidget(self.dend_len_spinbox)
        morph_layout.addLayout(dend_len_layout)

        dend_diam_layout = QHBoxLayout()
        dend_diam_layout.addWidget(QLabel("Dendrite Diameter (μm):"))
        self.dend_diam_spinbox = QDoubleSpinBox()
        self.dend_diam_spinbox.setRange(0.1, 100.0)
        self.dend_diam_spinbox.setSingleStep(0.1)
        self.dend_diam_spinbox.setValue(2.0)
        dend_diam_layout.addWidget(self.dend_diam_spinbox)
        morph_layout.addLayout(dend_diam_layout)

        axon_len_layout = QHBoxLayout()
        axon_len_layout.addWidget(QLabel("Axon Length (μm):"))
        self.axon_len_spinbox = QDoubleSpinBox()
        self.axon_len_spinbox.setRange(1.0, 10000.0)
        self.axon_len_spinbox.setSingleStep(10.0)
        self.axon_len_spinbox.setValue(1000.0)
        axon_len_layout.addWidget(self.axon_len_spinbox)
        morph_layout.addLayout(axon_len_layout)

        axon_diam_layout = QHBoxLayout()
        axon_diam_layout.addWidget(QLabel("Axon Diameter (μm):"))
        self.axon_diam_spinbox = QDoubleSpinBox()
        self.axon_diam_spinbox.setRange(0.1, 100.0)
        self.axon_diam_spinbox.setSingleStep(0.1)
        self.axon_diam_spinbox.setValue(1.0)
        axon_diam_layout.addWidget(self.axon_diam_spinbox)
        morph_layout.addLayout(axon_diam_layout)

        num_dend_layout = QHBoxLayout()
        num_dend_layout.addWidget(QLabel("Number of Dendrites:"))
        self.num_dend_spinbox = QSpinBox()
        self.num_dend_spinbox.setRange(1, 20)
        self.num_dend_spinbox.setValue(3)
        num_dend_layout.addWidget(self.num_dend_spinbox)
        morph_layout.addLayout(num_dend_layout)

        uniform_layout.addWidget(morph_params_group)

        # Synapse parameters
        syn_params_group = QGroupBox("Synaptic Parameters")
        syn_layout = QVBoxLayout(syn_params_group)

        tau_rise_layout = QHBoxLayout()
        tau_rise_layout.addWidget(QLabel("Tau Rise (ms):"))
        self.tau_rise_spinbox = QDoubleSpinBox()
        self.tau_rise_spinbox.setRange(0.01, 100.0)
        self.tau_rise_spinbox.setSingleStep(0.1)
        self.tau_rise_spinbox.setValue(0.5)
        tau_rise_layout.addWidget(self.tau_rise_spinbox)
        syn_layout.addLayout(tau_rise_layout)

        tau_decay_layout = QHBoxLayout()
        tau_decay_layout.addWidget(QLabel("Tau Decay (ms):"))
        self.tau_decay_spinbox = QDoubleSpinBox()
        self.tau_decay_spinbox.setRange(0.01, 100.0)
        self.tau_decay_spinbox.setSingleStep(0.1)
        self.tau_decay_spinbox.setValue(5.0)
        tau_decay_layout.addWidget(self.tau_decay_spinbox)
        syn_layout.addLayout(tau_decay_layout)

        uniform_layout.addWidget(syn_params_group)

        layout.addWidget(self.uniform_group)

        # table of parameters per neuron
        self.per_neuron_group = QGroupBox("Per-Neuron Parameters")
        per_neuron_layout = QVBoxLayout(self.per_neuron_group)

        info_label = QLabel(
            "Configure individual neuron parameters. Rows will be created based on network size.")
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        info_label.setWordWrap(True)
        per_neuron_layout.addWidget(info_label)

        self.neuron_params_table = QTableWidget()
        self.neuron_params_table.setColumnCount(11)
        self.neuron_params_table.setHorizontalHeaderLabels([
            "Neuron", "dt", "Temp", "Threshold", "Refrac",
            "Dend Len", "Dend Diam", "Axon Len", "Axon Diam",
            "Tau Rise", "Tau Decay"
        ])
        self.neuron_params_table.horizontalHeader().setStretchLastSection(True)
        self.neuron_params_table.setRowCount(0)
        per_neuron_layout.addWidget(self.neuron_params_table)

        layout.addWidget(self.per_neuron_group)

        layout.addStretch()

        # Initialize visibility
        self.on_mode_changed(True)

    def on_mode_changed(self, uniform):
        self.uniform_group.setVisible(uniform)
        self.per_neuron_group.setVisible(not uniform)

    def update_table_size(self, num_neurons):
        if not self.uniform_mode.isChecked():
            current_rows = self.neuron_params_table.rowCount()
            if current_rows < num_neurons:
                # Add rows
                for i in range(current_rows, num_neurons):
                    row = self.neuron_params_table.rowCount()
                    self.neuron_params_table.insertRow(row)

                    defaults = [
                        str(i), "0.01", "37.0", "0.0", "50.0",
                        "200.0", "2.0", "1000.0", "1.0", "0.5", "5.0"
                    ]
                    for col, default in enumerate(defaults):
                        item = QTableWidgetItem(default)
                        if col == 0:
                            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        self.neuron_params_table.setItem(row, col, item)
            elif current_rows > num_neurons:
                for i in range(current_rows - 1, num_neurons - 1, -1):
                    self.neuron_params_table.removeRow(i)

    def get_parameters(self):
        if self.uniform_mode.isChecked():
            # Return None to use default uniform parameters
            return None
        else:
            # Return list of parameter dicts
            params_list = []
            for row in range(self.neuron_params_table.rowCount()):
                try:
                    params = {
                        'dt': float(self.neuron_params_table.item(row, 1).text()),
                        'temperature': float(self.neuron_params_table.item(row, 2).text()),
                        'spike_threshold': float(self.neuron_params_table.item(row, 3).text()),
                        'refractory_period': float(self.neuron_params_table.item(row, 4).text()),
                        'default_dendrite_length': float(self.neuron_params_table.item(row, 5).text()),
                        'default_dendrite_diameter': float(self.neuron_params_table.item(row, 6).text()),
                        'default_axon_length': float(self.neuron_params_table.item(row, 7).text()),
                        'default_axon_diameter': float(self.neuron_params_table.item(row, 8).text()),
                        'default_tau_rise': float(self.neuron_params_table.item(row, 9).text()),
                        'default_tau_decay': float(self.neuron_params_table.item(row, 10).text()),
                    }
                    params_list.append(params)
                except:
                    # Skip invalid rows
                    continue

            return params_list if params_list else None
