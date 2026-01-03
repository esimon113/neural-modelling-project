
import numpy as np
from typing import Optional
from dataclasses import dataclass
import random

from neuron import Neuron
from parameters import NeuronParameters, SynapticParameters


@dataclass
class Connection:
    """
    Represents a synaptic connection between two neurons.

    Parameters:
        `pre_neuron_id`:       Presynaptic neuron ID
        `post_neuron_id`:      Postsynaptic neuron ID
        `post_compartment_id`: Postsynaptic compartment ID
        `synapse_index`:       Index of synapse within the postsynaptic compartment
        `weight`:              Connection weight (scaling factor)
        `delay`:               Synaptic delay (ms)
    """

    pre_neuron_id: int
    post_neuron_id: int
    post_compartment_id: int
    synapse_index: int
    weight: float
    delay: float


class NeuronalNetwork:
    """
    This class manages neurons and their synaptic connections, providing methods for simulation and analysis.
    """

    def __init__(self, num_neurons: int = 5, neuron_params: Optional[NeuronParameters] = None, neuron_params_list: Optional[list[NeuronParameters]] = None):
        """ 
        Args:
            `num_neurons`:        Number of neurons in the network
            `neuron_params`:      Single parameter set for all neurons (used if neuron_params_list is None)
            `neuron_params_list`: List of parameter sets for each neuron (overrides neuron_params if provided)
        """

        self.num_neurons = num_neurons

        # Handle parameter options
        if neuron_params_list is not None:
            if len(neuron_params_list) != num_neurons:
                raise ValueError(f"neuron_params_list length ({len(
                    neuron_params_list)}) must match num_neurons ({num_neurons})")
            self.neuron_params_list = neuron_params_list
        else:  # Use same params for all neurons
            default_params = neuron_params or NeuronParameters()
            self.neuron_params_list = [default_params] * num_neurons

        self.neurons: list[Neuron] = []
        for i in range(num_neurons):
            neuron = Neuron(i, self.neuron_params_list[i])
            # default neuron morphology with 3 dendrites with each 200μm length
            neuron.create_morphology(
                num_dendrites=3, dendrite_lengths=[200.0] * 3)
            self.neurons.append(neuron)

        self.connections: list[Connection] = []
        self.connection_matrix = np.zeros((num_neurons, num_neurons))

        self.t = 0.0
        # use dt from first neuron for network
        self.dt = self.neuron_params_list[0].dt

    def connect_neurons(self, pre_neuron_id: int, post_neuron_id: int, weight: float = 1.0, delay: float = 1.0, post_compartment_id: Optional[int] = None, synapse_params: Optional[SynapticParameters] = None) -> int:
        """
        Connect two neurons with a synaptic connection and returns the connection index.

        Args:
            `pre_neuron_id`:       Presynaptic neuron ID
            `post_neuron_id`:      Postsynaptic neuron ID
            `weight`:              Connection weight (scaling factor)
            `delay`:               Synaptic delay (ms)
            `post_compartment_id`: Postsynaptic compartment ID (uses first dendrite if None)
            `synapse_params`:      Synaptic parameters (uses defaults if None)
        """

        if not (0 <= pre_neuron_id < self.num_neurons and 0 <= post_neuron_id < self.num_neurons):
            raise ValueError("Invalid neuron IDs")

        if post_compartment_id is None:
            post_compartment_id = self.neurons[post_neuron_id].dendrite_ids[0]

        if synapse_params is None:
            synapse_params = SynapticParameters()

        self.neurons[post_neuron_id].add_synapse(post_compartment_id, synapse_params.tau_rise,
                                                 synapse_params.tau_decay, synapse_params.amplitude * weight, synapse_params.reversal_potential)
        synapse_index = len(
            self.neurons[post_neuron_id].compartments[post_compartment_id].synapses) - 1

        connection = Connection(pre_neuron_id=pre_neuron_id, post_neuron_id=post_neuron_id,
                                post_compartment_id=post_compartment_id, synapse_index=synapse_index, weight=weight, delay=delay)
        self.connections.append(connection)
        self.connection_matrix[pre_neuron_id, post_neuron_id] = weight

        return len(self.connections) - 1

    def add_external_input(self, neuron_id: int, compartment_id: int, t_start: float, t_end: float, amplitude: float):
        """
        Add external current input to a specific compartment.

        Args:
            `neuron_id`:      Target neuron ID
            `compartment_id`: Target compartment ID
            `t_start`:        Start time (ms)
            `t_end`:          End time (ms)
            `amplitude`:      Current amplitude (nA)
        """

        if not (0 <= neuron_id < self.num_neurons):
            raise ValueError("Invalid neuron ID")

        if not hasattr(self, 'external_inputs'):
            self.external_inputs = []

        self.external_inputs.append({
            'neuron_id': neuron_id,
            'compartment_id': compartment_id,
            't_start': t_start,
            't_end': t_end,
            'amplitude': amplitude
        })

    # TODO: implement multithreading for larger simulations?!

    def simulate(self, duration: float):
        """
        Simulate the network for a specified duration.

        Args:
            `duration`: Simulation duration (ms)
        """

        steps = int(duration / self.dt)

        if not hasattr(self, 'external_inputs'):
            self.external_inputs = []

        for step in range(steps):
            current_time = step * self.dt

            external_currents = {}
            for input_info in self.external_inputs:
                if input_info['t_start'] <= current_time < input_info['t_end']:
                    neuron_id = input_info['neuron_id']
                    compartment_id = input_info['compartment_id']
                    amplitude = input_info['amplitude']

                    if neuron_id not in external_currents:
                        external_currents[neuron_id] = {}
                    external_currents[neuron_id][compartment_id] = amplitude

            for neuron_id, neuron in enumerate(self.neurons):  # Update neurons
                I_ext = external_currents.get(neuron_id, {})
                neuron.update(I_ext)

                if neuron.has_soma_spiked():
                    self._propagate_spike(neuron_id, current_time)

            self.t += self.dt

    def _propagate_spike(self, pre_neuron_id: int, spike_time: float):
        for connection in self.connections:
            if connection.pre_neuron_id == pre_neuron_id:
                # add delay
                delayed_spike_time = spike_time + connection.delay
                # Add spike to post-synapse
                self.neurons[connection.post_neuron_id].add_spike_to_synapse(
                    connection.post_compartment_id, connection.synapse_index, delayed_spike_time)

    def get_network_state(self) -> dict:
        state = {
            'time': self.t,
            'neurons': []
        }

        for i, neuron in enumerate(self.neurons):
            neuron_state = {
                'id': i,
                'soma_voltage': neuron.get_soma_voltage(),
                'axon_voltage': neuron.get_axon_voltage(),
                'compartments': []
            }

            for j, compartment in enumerate(neuron.compartments):
                compartment_state = {
                    'id': j,
                    'type': compartment.type.value,
                    'voltage': compartment.state.V,
                    'spike_count': len(compartment.spike_times)
                }
                neuron_state['compartments'].append(compartment_state)

            state['neurons'].append(neuron_state)

        return state

    def get_voltage_histories(self) -> dict[int, dict[int, np.ndarray]]:
        traces = {}
        for i, neuron in enumerate(self.neurons):
            traces[i] = neuron.get_voltage_histories()

        return traces

    def get_spike_times(self) -> dict[int, dict[int, list[float]]]:
        spikes = {}
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.get_spike_times()

        return spikes

    def get_network_statistics(self) -> dict:
        stats = {
            'num_neurons': self.num_neurons,
            'num_connections': len(self.connections),
            'connection_density': len(self.connections) / (self.num_neurons * (self.num_neurons - 1)),
            'neuron_stats': []
        }

        for i, neuron in enumerate(self.neurons):
            neuron_spikes = neuron.get_spike_times()
            soma_spikes = neuron_spikes.get(neuron.soma_id, [])
            axon_spikes = neuron_spikes.get(neuron.axon_id, [])

            neuron_stat = {
                'id': i,
                'soma_spikes': len(soma_spikes),
                'axon_spikes': len(axon_spikes),
                # ms -> s
                'soma_firing_rate': len(soma_spikes) / (self.t / 1000.0) if self.t > 0 else 0,
                'axon_firing_rate': len(axon_spikes) / (self.t / 1000.0) if self.t > 0 else 0
            }
            stats['neuron_stats'].append(neuron_stat)

        return stats

    def reset_to_initial_state(self):
        for neuron in self.neurons:
            neuron.reset_to_initial_state()

        self.t = 0.0

        if hasattr(self, 'external_inputs'):
            self.external_inputs = []

    def clear_network(self):
        num_connections = len(self.connections)
        self.connections.clear()
        self.connection_matrix.fill(0.0)

        if hasattr(self, 'external_inputs'):
            self.external_inputs.clear()

    def create_feedforward_network(self, layer_sizes: list[int], connection_weights: list[float] = None, clear_existing: bool = True) -> list[list[int]]:
        """
        Create a feedforward network architecture and returns a list of neuron ids in each layer.

        Args:
            `layer_sizes`:        List of layer sizes (i.e., [2, 3, 1] for a 2-3-1 network)
            `connection_weights`: List of weights for connections between layers. If None, uses 1.0 for all. If shorter than needed, repeats the last value. If longer, truncates.
            `clear_existing`:     Whether to clear existing connections before creating new ones
        """

        if sum(layer_sizes) != self.num_neurons:
            raise ValueError(f"Total layer size ({sum(
                layer_sizes)}) must equal num_neurons ({self.num_neurons})")

        if clear_existing:
            self.clear_network()

        # Assign neurons to layers
        layers = []
        neuron_idx = 0
        for layer_size in layer_sizes:
            layer = list(range(neuron_idx, neuron_idx + layer_size))
            layers.append(layer)
            neuron_idx += layer_size

        total_connections = 0
        for i in range(len(layers) - 1):
            total_connections += len(layers[i]) * len(layers[i + 1])

        # Handle connection weights
        if connection_weights is None:
            connection_weights = [1.0] * total_connections
        elif len(connection_weights) < total_connections:  # extend if too short
            last_weight = connection_weights[-1] if connection_weights else 1.0
            connection_weights.extend(
                [last_weight] * (total_connections - len(connection_weights)))
        elif len(connection_weights) > total_connections:  # truncate if too many
            connection_weights = connection_weights[:total_connections]

        # connect layers with weights
        weight_idx = 0
        for i in range(len(layers) - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]

            for pre_neuron in current_layer:
                for post_neuron in next_layer:
                    self.connect_neurons(
                        pre_neuron, post_neuron, connection_weights[weight_idx])
                    weight_idx += 1

        return layers

    def create_random_network(self, connection_probability: float = 0.3, weight_range: tuple[float, float] = (0.5, 2.0), clear_existing: bool = True) -> int:
        """
        Create random connections between neurons and returns the number of connections created.

        Args:
            `connection_probability`: Probability of connection between any two neurons
            `weight_range`:           Range for random connection weights
            `clear_existing`:         Whether to clear existing connections before creating new ones
        """

        if clear_existing:
            self.clear_network()

        connections_created = 0
        for pre_neuron in range(self.num_neurons):
            for post_neuron in range(self.num_neurons):
                if pre_neuron != post_neuron and random.random() < connection_probability:
                    weight = random.uniform(weight_range[0], weight_range[1])
                    self.connect_neurons(pre_neuron, post_neuron, weight)
                    connections_created += 1

        return connections_created

# TODO: maybe provide more network topologies?
