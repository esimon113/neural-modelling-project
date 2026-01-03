
import numpy as np
from network import NeuronalNetwork
from parameters import NeuronParameters


# TODO: Make more user friendly


def create_neuron_types():
    # Input layer:
    input_params0 = NeuronParameters(
        dt=0.01,
        temperature=37.0,
        spike_threshold=-5.0,
        refractory_period=60.0,
        default_tau_rise=0.2,
        default_tau_decay=3.0,
        default_dendrite_length=150.0,
        default_dendrite_diameter=1.0,
        default_axon_length=500.0,
        default_axon_diameter=0.8
    )

    input_params1 = NeuronParameters(
        dt=0.01,
        temperature=37.0,
        spike_threshold=-7.0,
        refractory_period=50.0,
        default_tau_rise=0.18,
        default_tau_decay=3.5,
        default_dendrite_length=200.0,
        default_dendrite_diameter=1.6,
        default_axon_length=800.0,
        default_axon_diameter=1.1
    )

    # Hidden layer 1:
    hidden1_params0 = NeuronParameters(
        dt=0.01,
        temperature=37.0,
        spike_threshold=0.0,
        refractory_period=20.0,
        default_tau_rise=0.3,
        default_tau_decay=4.0,
        default_dendrite_length=200.0,
        default_dendrite_diameter=1.5,
        default_axon_length=600.0,
        default_axon_diameter=1.0
    )
    hidden1_params1 = NeuronParameters(
        dt=0.01,
        temperature=36.5,
        spike_threshold=-1.0,
        refractory_period=25.0,
        default_tau_rise=0.4,
        default_tau_decay=4.1,
        default_dendrite_length=220.0,
        default_dendrite_diameter=1.55,
        default_axon_length=670.0,
        default_axon_diameter=1.05
    )
    hidden1_params2 = NeuronParameters(
        dt=0.01,
        temperature=37.0,
        spike_threshold=0.0,
        refractory_period=22.0,
        default_tau_rise=0.35,
        default_tau_decay=4.4,
        default_dendrite_length=230.0,
        default_dendrite_diameter=1.65,
        default_axon_length=690.0,
        default_axon_diameter=1.2
    )

    # Hidden layer 2:
    hidden2_params0 = NeuronParameters(
        dt=0.01,
        temperature=37.0,
        spike_threshold=5.0,
        refractory_period=30.0,
        default_tau_rise=0.5,
        default_tau_decay=5.0,
        default_dendrite_length=250.0,
        default_dendrite_diameter=2.0,
        default_axon_length=700.0,
        default_axon_diameter=1.2
    )
    hidden2_params1 = NeuronParameters(
        dt=0.01,
        temperature=36.8,
        spike_threshold=8.0,
        refractory_period=40.0,
        default_tau_rise=0.6,
        default_tau_decay=4.6,
        default_dendrite_length=320.0,
        default_dendrite_diameter=2.5,
        default_axon_length=810.0,
        default_axon_diameter=1.4
    )

    # Output layer:
    output_params = NeuronParameters(
        dt=0.01,
        temperature=37.0,
        spike_threshold=0.0,
        refractory_period=45.0,
        default_tau_rise=0.4,
        default_tau_decay=5.2,
        default_dendrite_length=310.0,
        default_dendrite_diameter=1.8,
        default_axon_length=920.0,
        default_axon_diameter=1.1
    )

    neuron_params_list = [
        input_params0, input_params1,
        hidden1_params0, hidden1_params1, hidden1_params2,
        hidden2_params0, hidden2_params1,
        output_params
    ]

    return neuron_params_list


def create_network():
    neuron_params = create_neuron_types()
    network = NeuronalNetwork(num_neurons=8, neuron_params_list=neuron_params)
    layers = network.create_feedforward_network(
        [2, 3, 2, 1], clear_existing=True)
    network.clear_network()

    # Format: (pre, post, weight, synaptic delay)
    weight_matrix = [
        (0, 2, 1.5, 0.43),
        (1, 3, 1.2, 0.51),
        (1, 4, 0.8, 0.52),
        (1, 6, 0.3, 0.55),
        (2, 3, 1.0, 0.48),
        (2, 5, 0.9, 0.5),
        (3, 5, 0.4, 0.45),
        (3, 6, 0.2, 0.51),
        (4, 6, 0.6, 0.38),
        (5, 7, 1.8, 0.52),
        (6, 7, 0.5, 0.6),
    ]

    for pre_id, post_id, weight, delay in weight_matrix:
        network.connect_neurons(
            pre_neuron_id=pre_id, post_neuron_id=post_id, weight=weight, delay=delay)

    return network, layers
