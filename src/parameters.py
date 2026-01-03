
from enum import Enum
from typing import Optional
import numpy as np


class CompartmentType(Enum):
    """Types of neuronal compartments."""
    SOMA = "soma"
    AXON = "axon"
    DENDRITE = "dendrite"


class CompartmentParameters:
    """
    Parameters for a single neuronal compartment.
    This class defines the morphological, electrical properties of a compartment, etc.
    The units are listed below:

    Parameters:
        `length`:   Length in μm
        `diameter`: Diameter in μm
        `area`:     Surface area in μm² (calculated if None)
        `C_m`:      Membrane capacitance (μF/cm²)
        `R_m`:      Membrane resistance (Ω·cm²)
        `R_a`:      Axial resistance (Ω·cm)
        `g_Na`:     Sodium conductance (mS/cm²)
        `g_K`:      Potassium conductance (mS/cm²)
        `g_L`:      Leak conductance (mS/cm²)
        `E_Na`:     Sodium reversal potential (mV)
        `E_K`:      Potassium reversal potential (mV)
        `E_L`:      Leak reversal potential (mV)
        `g_Ca`:     Calcium conductance (mS/cm²)
        `g_K_Ca`:   Calcium-activated potassium conductance (mS/cm²)
        `g_h`:      Hyperpolarization-activated conductance (mS/cm²)
        `g_Kv`:     Voltage-gated potassium conductance (mS/cm²)
        `E_Ca`:     Calcium reversal potential (mV)
        `tau_Ca`:   Calcium decay time constant (ms)
        `Ca_0`:     Initial calcium concentration (μM)
        `refractory_period`: Refractory period (ms)
        `spike_threshold`: Spike detection threshold (mV)
    """

    # Morphological properties
    length: float = 100.0
    diameter: float = 2.0
    area: Optional[float] = None
    # Electrical properties
    C_m: float = 1.0
    R_m: float = 20000.0
    R_a: float = 100.0
    # Hodgkin-Huxley channel conductances (mS/cm²)
    g_Na: float = 120.0
    g_K: float = 36.0
    g_L: float = 0.3
    # Reversal potentials (mV)
    E_Na: float = 50.0
    E_K: float = -77.0
    E_L: float = -54.387
    # Dendritic-specific channels (mS/cm²)
    g_Ca: float = 0.0
    g_K_Ca: float = 0.0
    g_h: float = 0.0
    g_Kv: float = 0.0
    # Calcium dynamics
    E_Ca: float = 120.0
    tau_Ca: float = 20.0
    Ca_0: float = 0.0

    # Refractory period
    refractory_period: float = 10.0

    # Spike detection
    spike_threshold: float = 0.0

    def __post_init__(self):
        """Calculates compartmetn surface area (if not set)."""
        if self.area is None:
            self.area = np.pi * self.diameter * self.length

        self.area_cm2 = self.area * 1e-8  # µm² → cm²


class NeuronParameters:
    """
    Params for a complete multi-compartment neuron.

    Parameters:
        `dt`:                         Time step (ms)
        `temperature`:                Temperature (°C)
        `spike_threshold`:            Spike detection threshold (mV)
        `refractory_period`:          Refractory period (ms)
        `default_tau_rise`:           Default synaptic rise time (ms)
        `default_tau_decay`:          Default synaptic decay time (ms)
        `default_dendrite_length`:    Default dendrite length (μm)
        `default_dendrite_diameter`:  Default dendrite diameter (μm)
        `default_axon_length`:        Default axon length (μm)
        `default_axon_diameter`:      Default axon diameter (μm)
    """

    # Simulation params
    dt: float = 0.01
    temperature: float = 37.0

    spike_threshold: float = 0.0
    refractory_period: float = 50.0

    # default compartment params
    default_tau_rise: float = 0.5
    default_tau_decay: float = 5.0
    default_dendrite_length: float = 200.0
    default_dendrite_diameter: float = 2.0
    default_axon_length: float = 1000.0
    default_axon_diameter: float = 1.0


class SynapticParameters:
    """
    Params for connections between neurons.

    Note: `plasticity_enabled` and `learning_rate` are not yet implemented.

    Parameters:
        `tau_rise`:             Rise time constant (ms)
        `tau_decay`:            Decay time constant (ms)
        `amplitude`:            Peak conductance (nS)
        `reversal_potential`:   Reversal potential (mV)
        `delay`:                Synaptic delay (ms)
        `weight`:               Connection weight (scaling factor)
        `plasticity_enabled`:   Whether plasticity is enabled
        `learning_rate`:        Learning rate
    """

    # Synapse dynamics
    tau_rise: float = 0.5
    tau_decay: float = 5.0
    amplitude: float = 1.0
    reversal_potential: float = 0.0

    # Connection
    delay: float = 1.0
    weight: float = 1.0

    # TODO: Add Plasticity
    plasticity_enabled: bool = False
    learning_rate: float = 0.01


# Predefined parameter sets for compartments
SOMA_PARAMETERS = CompartmentParameters(  # use hodgkin-huxley vals
    length=20.0,
    diameter=20.0,
    g_Na=120.0,
    g_K=36.0,
    g_L=0.3,
    C_m=1.0
)

AXON_PARAMETERS = CompartmentParameters(
    length=1000.0,
    diameter=1.0,
    g_Na=200.0,
    g_K=50.0,
    g_L=0.1,
    C_m=0.5
)

DENDRITE_PARAMETERS = CompartmentParameters(
    length=200.0,
    diameter=2.0,
    g_Na=20.0,
    g_K=10.0,
    g_L=0.3,
    g_Ca=2.0,
    g_K_Ca=5.0,
    g_h=1.0,
    g_Kv=3.0,
    C_m=1.0
)
