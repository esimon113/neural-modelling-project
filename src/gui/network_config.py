from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QSpinBox, QDoubleSpinBox, QComboBox, QPushButton,
                             QTableWidget, QTableWidgetItem,
                             QGroupBox, QLineEdit)


class NetworkConfigWidget(QWidget):
    """Widget for configuring network parameters."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Number of neurons
        neurons_layout = QHBoxLayout()
        neurons_layout.addWidget(QLabel("Number of Neurons:"))
        self.num_neurons_spinbox = QSpinBox()
        self.num_neurons_spinbox.setRange(1, 100)
        self.num_neurons_spinbox.setValue(8)
        self.num_neurons_spinbox.valueChanged.connect(
            self.on_num_neurons_changed)
        neurons_layout.addWidget(self.num_neurons_spinbox)
        neurons_layout.addStretch()
        layout.addLayout(neurons_layout)

        # Network topology
        topology_group = QGroupBox("Network Topology")
        topology_layout = QVBoxLayout(topology_group)

        topology_layout.addWidget(QLabel("Type:"))
        self.topology_combo = QComboBox()
        self.topology_combo.addItems(["feedforward", "random", "custom"])
        self.topology_combo.currentTextChanged.connect(
            self.on_topology_changed)
        topology_layout.addWidget(self.topology_combo)

        # Feedforward settings
        self.feedforward_group = QGroupBox("Feedforward Settings")
        ff_layout = QVBoxLayout(self.feedforward_group)

        ff_layout.addWidget(QLabel("Layer Sizes (comma-separated):"))
        self.layer_sizes_input = QLineEdit()
        self.layer_sizes_input.setText("2, 3, 2, 1")
        self.layer_sizes_input.setPlaceholderText("e.g., 2, 3, 2, 1")
        ff_layout.addWidget(self.layer_sizes_input)
        topology_layout.addWidget(self.feedforward_group)

        # Random network settings
        self.random_group = QGroupBox("Random Network Settings")
        random_layout = QVBoxLayout(self.random_group)

        prob_layout = QHBoxLayout()
        prob_layout.addWidget(QLabel("Connection Probability:"))
        self.connection_prob_spinbox = QDoubleSpinBox()
        self.connection_prob_spinbox.setRange(0.0, 1.0)
        self.connection_prob_spinbox.setSingleStep(0.1)
        self.connection_prob_spinbox.setValue(0.3)
        prob_layout.addWidget(self.connection_prob_spinbox)
        random_layout.addLayout(prob_layout)

        weight_min_layout = QHBoxLayout()
        weight_min_layout.addWidget(QLabel("Weight Range Min:"))
        self.weight_min_spinbox = QDoubleSpinBox()
        self.weight_min_spinbox.setRange(0.0, 10.0)
        self.weight_min_spinbox.setSingleStep(0.1)
        self.weight_min_spinbox.setValue(0.5)
        weight_min_layout.addWidget(self.weight_min_spinbox)
        random_layout.addLayout(weight_min_layout)

        weight_max_layout = QHBoxLayout()
        weight_max_layout.addWidget(QLabel("Weight Range Max:"))
        self.weight_max_spinbox = QDoubleSpinBox()
        self.weight_max_spinbox.setRange(0.0, 10.0)
        self.weight_max_spinbox.setSingleStep(0.1)
        self.weight_max_spinbox.setValue(2.0)
        weight_max_layout.addWidget(self.weight_max_spinbox)
        random_layout.addLayout(weight_max_layout)
        topology_layout.addWidget(self.random_group)

        # Custom connections table
        self.custom_group = QGroupBox("Custom Connections")
        custom_layout = QVBoxLayout(self.custom_group)

        custom_info = QLabel("Format: (pre, post, weight, delay)")
        custom_info.setStyleSheet("color: #666; font-size: 10px;")
        custom_layout.addWidget(custom_info)

        self.connections_table = QTableWidget()
        self.connections_table.setColumnCount(4)
        self.connections_table.setHorizontalHeaderLabels(
            ["Pre", "Post", "Weight", "Delay"])
        self.connections_table.horizontalHeader().setStretchLastSection(True)
        self.connections_table.setRowCount(0)
        custom_layout.addWidget(self.connections_table)

        add_conn_layout = QHBoxLayout()
        self.add_conn_button = QPushButton("Add Connection")
        self.add_conn_button.clicked.connect(self.add_connection_row)
        add_conn_layout.addWidget(self.add_conn_button)

        self.remove_conn_button = QPushButton("Remove Selected")
        self.remove_conn_button.clicked.connect(self.remove_connection_row)
        add_conn_layout.addWidget(self.remove_conn_button)
        custom_layout.addLayout(add_conn_layout)

        topology_layout.addWidget(self.custom_group)
        layout.addWidget(topology_group)

        # External inputs
        external_group = QGroupBox("External Inputs")
        external_layout = QVBoxLayout(external_group)

        self.external_inputs_table = QTableWidget()
        self.external_inputs_table.setColumnCount(5)
        self.external_inputs_table.setHorizontalHeaderLabels(
            ["Neuron", "Compartment",
                "Start (ms)", "End (ms)", "Amplitude (nA)"]
        )
        self.external_inputs_table.horizontalHeader().setStretchLastSection(True)
        self.external_inputs_table.setRowCount(0)
        external_layout.addWidget(self.external_inputs_table)

        ext_buttons_layout = QHBoxLayout()
        self.add_ext_button = QPushButton("Add Input")
        self.add_ext_button.clicked.connect(self.add_external_input_row)
        ext_buttons_layout.addWidget(self.add_ext_button)

        self.remove_ext_button = QPushButton("Remove Selected")
        self.remove_ext_button.clicked.connect(self.remove_external_input_row)
        ext_buttons_layout.addWidget(self.remove_ext_button)
        external_layout.addLayout(ext_buttons_layout)

        layout.addWidget(external_group)

        layout.addStretch()

        # Initialize visibility
        self.on_topology_changed("feedforward")

    def on_num_neurons_changed(self, value):
        """NOT YET IMPLEMENTED!"""
        pass

    def on_topology_changed(self, topology):
        self.feedforward_group.setVisible(topology == "feedforward")
        self.random_group.setVisible(topology == "random")
        self.custom_group.setVisible(topology == "custom")

    def add_connection_row(self):
        row = self.connections_table.rowCount()
        self.connections_table.insertRow(row)

        # Set default values
        for col in range(4):
            item = QTableWidgetItem("0")
            self.connections_table.setItem(row, col, item)

    def remove_connection_row(self):
        current_row = self.connections_table.currentRow()
        if current_row >= 0:
            self.connections_table.removeRow(current_row)

    def add_external_input_row(self):
        row = self.external_inputs_table.rowCount()
        self.external_inputs_table.insertRow(row)

        defaults = ["0", "0", "5.0", "15.0", "0.05"]
        for col, default in enumerate(defaults):
            item = QTableWidgetItem(default)
            self.external_inputs_table.setItem(row, col, item)

    def remove_external_input_row(self):
        current_row = self.external_inputs_table.currentRow()
        if current_row >= 0:
            self.external_inputs_table.removeRow(current_row)

    def get_parameters(self):
        num_neurons = self.num_neurons_spinbox.value()
        topology = self.topology_combo.currentText()

        params = {
            'num_neurons': num_neurons,
            'topology': topology
        }

        if topology == "feedforward":
            # Parse layer sizes
            try:
                layer_sizes_str = self.layer_sizes_input.text()
                layer_sizes = [int(x.strip())
                               for x in layer_sizes_str.split(',')]
                if sum(layer_sizes) != num_neurons:
                    raise ValueError(f"Layer sizes sum ({sum(
                        layer_sizes)}) must equal number of neurons ({num_neurons})")
                params['layer_sizes'] = layer_sizes
            except Exception as e:
                raise ValueError(f"Invalid layer sizes: {str(e)}")

        elif topology == "random":
            params['connection_probability'] = self.connection_prob_spinbox.value()
            params['weight_range'] = (
                self.weight_min_spinbox.value(),
                self.weight_max_spinbox.value()
            )

        elif topology == "custom":
            # Get connections from table
            connections = []
            for row in range(self.connections_table.rowCount()):
                try:
                    pre = int(self.connections_table.item(row, 0).text())
                    post = int(self.connections_table.item(row, 1).text())
                    weight = float(self.connections_table.item(row, 2).text())
                    delay = float(self.connections_table.item(row, 3).text())
                    connections.append((pre, post, weight, delay))
                except:
                    continue
            params['connections'] = connections

        # Get external inputs
        external_inputs = []
        for row in range(self.external_inputs_table.rowCount()):
            try:
                neuron_id = int(self.external_inputs_table.item(row, 0).text())
                compartment_id = int(
                    self.external_inputs_table.item(row, 1).text())
                t_start = float(self.external_inputs_table.item(row, 2).text())
                t_end = float(self.external_inputs_table.item(row, 3).text())
                amplitude = float(
                    self.external_inputs_table.item(row, 4).text())
                external_inputs.append({
                    'neuron_id': neuron_id,
                    'compartment_id': compartment_id,
                    't_start': t_start,
                    't_end': t_end,
                    'amplitude': amplitude
                })
            except:
                continue

        params['external_inputs'] = external_inputs

        return params
