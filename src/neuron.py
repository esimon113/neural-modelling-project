
import numpy as np
from typing import Optional
import copy

from compartment import Compartment, Synapse
from parameters import CompartmentParameters, NeuronParameters, CompartmentType, SOMA_PARAMETERS, AXON_PARAMETERS, DENDRITE_PARAMETERS


class Neuron:
    """
    Multi-compartment neuron with somewhat realistic morphology.

    This class implements a neuron consisting of:
    - Soma
    - Axon
    - Variable number of dendrites
    """

    def __init__(self, neuron_id: int, params: Optional[NeuronParameters] = None):
        """
        Initialize a multi-compartment neuron.

        Args:
            `neuron_id`: Unique identifier for this neuron
            `params`:    Neuron parameters (uses defaults if None)
        """

        self.id = neuron_id
        self.params = params or NeuronParameters()

        self.compartments: list[Compartment] = []
        self.soma_id: Optional[int] = None
        self.axon_id: Optional[int] = None
        self.dendrite_ids: list[int] = []

        self.connections: dict[int, list[int]] = {}
        self.t = 0.0

    def add_compartment(self, comp_type: CompartmentType, params: Optional[CompartmentParameters] = None) -> int:
        """
        Returns the added compartsments id.

        Args:
            `comp_type`: Type of compartment to add
            `params`:    Compartment parameters (uses defaults if None)
        """

        comp_id = len(self.compartments)

        if params is None:
            if comp_type == CompartmentType.SOMA:
                params = SOMA_PARAMETERS
            elif comp_type == CompartmentType.AXON:
                params = AXON_PARAMETERS
            else:  # DENDRITE
                params = DENDRITE_PARAMETERS

        compartment = Compartment(comp_id, comp_type, params, self.params.dt)
        self.compartments.append(compartment)
        self.connections[comp_id] = []

        return comp_id

    def connect_compartments(self, comp1_id: int, comp2_id: int):
        if comp1_id != comp2_id:  # undirected -> direction comes from compartment type
            self.connections[comp1_id].append(comp2_id)
            self.connections[comp2_id].append(comp1_id)

    def create_morphology(self, num_dendrites: int = 3, dendrite_lengths: list[float] = None) -> tuple[int, int, list[int]]:
        """
        Create a complete neuron morphology containing soma, axon and dendrites.

        Args:
            `num_dendrites`:    Number of dendrites to create
            `dendrite_lengths`: List of dendrite lengths (μm). If None, uses default length for all. If shorter than num_dendrites, repeats the last value.

        Returns:
            Tuple of (soma_id, axon_id, dendrite_ids)
        """

        soma_params = copy.deepcopy(SOMA_PARAMETERS)
        soma_params.refractory_period = self.params.refractory_period
        soma_params.spike_threshold = self.params.spike_threshold
        self.soma_id = self.add_compartment(CompartmentType.SOMA, soma_params)
        axon_params = copy.deepcopy(AXON_PARAMETERS)
        axon_params.refractory_period = self.params.refractory_period
        axon_params.spike_threshold = self.params.spike_threshold
        self.axon_id = self.add_compartment(CompartmentType.AXON, axon_params)

        # Handle dendrite lengths
        if dendrite_lengths is None:  # default lengths
            dendrite_lengths = [200.0] * num_dendrites
        elif len(dendrite_lengths) < num_dendrites:  # extend if too short
            last_length = dendrite_lengths[-1] if dendrite_lengths else 200.0
            dendrite_lengths.extend(
                [last_length] * (num_dendrites - len(dendrite_lengths)))
        elif len(dendrite_lengths) > num_dendrites:  # truncate if too many
            dendrite_lengths = dendrite_lengths[:num_dendrites]

        self.dendrite_ids = []

        for i in range(num_dendrites):
            dendrite_params = copy.deepcopy(DENDRITE_PARAMETERS)
            dendrite_params.length = dendrite_lengths[i]
            dendrite_params.refractory_period = self.params.refractory_period
            dendrite_params.spike_threshold = self.params.spike_threshold
            dendrite_id = self.add_compartment(
                CompartmentType.DENDRITE, dendrite_params)
            self.dendrite_ids.append(dendrite_id)

        for dendrite_id in self.dendrite_ids:
            self.connect_compartments(self.soma_id, dendrite_id)

        self.connect_compartments(self.soma_id, self.axon_id)

        return self.soma_id, self.axon_id, self.dendrite_ids

    def calculate_axon_currents(self):
        for compartment in self.compartments:
            compartment.state.I_axial_in = 0.0
            compartment.state.I_axial_out = 0.0

        for comp_id, compartment in enumerate(self.compartments):
            for neighbor_id in self.connections[comp_id]:
                neighbor = self.compartments[neighbor_id]
                R_axial = (compartment.params.R_a * compartment.params.length +
                           neighbor.params.R_a * neighbor.params.length) / 2  # axial resistance
                I_axial = (neighbor.state.V - compartment.state.V) / \
                    R_axial  # Ohms law

                compartment.state.I_axial_in += I_axial
                neighbor.state.I_axial_out += I_axial

    def add_synapse(self, compartment_id: int, tau_rise: float = 0.5, tau_decay: float = 5.0, amplitude: float = 1.0, reversal_potential: float = 0.0) -> Synapse:
        """
        Add a synapse input to an compartment and returns the synapse object.

        Args:
            `compartment_id`:     id of compartment to which the synapse should be added
            `tau_rise`:           Synaptic rise time constant (ms)
            `tau_decay`:          Synaptic decay time constant (ms)
            `amplitude`:          Peak synaptic conductance (nS)
            `reversal_potential`: Synaptic reversal potential (mV)
        """

        # is compartment id valid?
        if not (0 <= compartment_id < len(self.compartments)):
            raise ValueError(f"Invalid compartment ID: {compartment_id}")

        synapse = Synapse(tau_rise, tau_decay, amplitude,
                          reversal_potential, self.params.dt)
        self.compartments[compartment_id].add_synapse(synapse)

        return synapse

    def update(self, I_ext: Optional[dict[int, float]] = None):
        """
        Args:
            `I_ext`: External current (nA) for each compartment, otherwise 0.0.
        """

        if I_ext is None:
            I_ext = {}

        self.calculate_axon_currents()

        for comp_id, compartment in enumerate(self.compartments):
            external_current = I_ext.get(comp_id, 0.0)
            compartment.update(external_current)

        for compartment in self.compartments:
            for synapse in compartment.synapses:
                synapse.update_state_time(self.params.dt)

        self.t += self.params.dt

    def add_spike_to_synapse(self, compartment_id: int, synapse_index: int, spike_time: float):
        """
        Args:
            `compartment_id`: id of compartment containing the synapse
            `synapse_index`:  Index of synapse within the compartment
            `spike_time`:     Time of spike (ms)
        """
        # validate compartment id and synapse index
        if (0 <= compartment_id < len(self.compartments) and 0 <= synapse_index < len(self.compartments[compartment_id].synapses)):
            self.compartments[compartment_id].synapses[synapse_index].add_spike(
                spike_time)

    def get_voltage_histories(self) -> dict[int, np.ndarray]: return {
        i: comp.get_voltage_history() for i, comp in enumerate(self.compartments)}

    def get_spike_times(self) -> dict[int, list[float]]: return {
        i: comp.get_spike_times() for i, comp in enumerate(self.compartments)}

    def get_soma_voltage(
        self) -> float: return 0.0 if self.soma_id is None else self.compartments[self.soma_id].state.V

    def get_axon_voltage(
        self) -> float: return 0.0 if self.axon_id is None else self.compartments[self.axon_id].state.V

    def has_soma_spiked(
        self) -> bool: return False if self.soma_id is None else self.compartments[self.soma_id].has_spiked()

    def has_axon_spiked(
        self) -> bool: return False if self.axon_id is None else self.compartments[self.axon_id].has_spiked()

    def reset_to_initial_state(self):
        for compartment in self.compartments:
            compartment.reset_to_init_state()

        self.t = 0.0

    def get_compartment_info(self) -> dict:
        info = {
            'soma_id': self.soma_id,
            'axon_id': self.axon_id,
            'dendrite_ids': self.dendrite_ids,
            'total_compartments': len(self.compartments),
            'compartments': []
        }

        for i, comp in enumerate(self.compartments):
            comp_info = {
                'id': i,
                'type': comp.type.value,
                'length': comp.params.length,
                'diameter': comp.params.diameter,
                'area': comp.params.area,
                'g_Na': comp.params.g_Na,
                'g_K': comp.params.g_K,
                'g_L': comp.params.g_L,
                'num_synapses': len(comp.synapses)
            }

            info['compartments'].append(comp_info)

        return info
