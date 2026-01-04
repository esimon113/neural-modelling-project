# Multi-Compartment Neuronal Network Model

A somewhat biologically realistic implementation of multi-compartment neurons based on cable theory, featuring Hodgkin-Huxley dynamics and synaptic modeling. This project provides a basic framework for simulating neuronal networks with detailed morphological and electrophysiological properties.

## Table of Contents

1. [Biological Background](#biological-background)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Architecture](#architecture)
4. [Installation and Usage](#installation-and-usage)
5. [Visualization](#visualization)
6. [GUI Wrapper](#gui-wrapper)
7. [References](#references)

## Biological Background

### Neurons and Neural Networks

Neurons are the fundamental computational units of the nervous system, responsible for processing and transmitting information through electrical and chemical signals. A typical neuron consists of:

- **Soma (Cell Body)**: Contains the nucleus and most organelles, integrates incoming signals
- **Dendrites**: Branching extensions that receive synaptic inputs from other neurons
- **Axon**: Long projection that transmits action potentials ("outputs") to target neurons
- **Synapses**: Specialized junctions where neurons communicate via neurotransmitters

### Action Potentials and Membrane Dynamics

The electrical activity of neurons is governed by the movement of ions across the cell membrane. The membrane potential (voltage difference across the membrane) changes dynamically due to:

1. **Ion Channels**: Proteins that allow specific ions to pass through the membrane
2. **Ion Pumps**: Active transporters that maintain ion concentration gradients
3. **Membrane Capacitance**: The membrane's ability to store electrical charge

### Multi-Compartment Models

Real (biological) neurons are not electrically uniform structures. Different parts of a neuron (soma, dendrites, axon) have different electrical properties and channel distributions. This allows for modeling multiple scales, as is illustrated by the following image taken from Dayan & Abbott (2001):

<img src="imgs/multi compartment model.png" alt="Scaling Multi-Compartment Neuron Model" width="400">


Multi-compartment models divide neurons into discrete segments, each with its own:

- Membrane capacitance and resistance
- Ion channel densities
- Morphological properties (length, diameter)
- Electrical coupling to neighboring compartments

This approach allows realistic modeling of:
- **Dendritic Integration**: How synaptic inputs from other neurons are processed in the dendrites
- **Action Potential Propagation**: How spikes travel along axons *(if an axon is modeled consisting of multiple compartments)*
- **Spatial Summation**: How inputs from different locations combine
- **Temporal Dynamics**: How signals change over time and space

### Cable Theory

Cable theory provides a mathematical foundation for understanding electrical signal propagation in neurons. An estimation is made by modeling dendrites and axons as cylinders *("cable")* composed of segments. So it basically treats neuronal processes as electrical cables with:

- **Axial Resistance ($R_a$)**: Resistance to current flow along the process
- **Membrane Resistance ($R_m$)**: Resistance to current flow across the membrane
- **Membrane Capacitance ($C_m$)**: Ability to store electrical charge

The cable equation describes how voltage changes propagate:

$$ c_m \frac{\partial V}{\partial t} = \frac{1}{2ar_L} \frac{\partial}{\partial x} \left( a^2 \frac{\partial V}{\partial x} \right) - i_m + i_e$$


Where:
- $c_m$ is membrane capacitance
- $V$ is membrane potential
- $a$ is cable radius
- $r_L$ is axial resistance (=longitudinal)
- $i_m$ is membrane current density
- $i_e$ is external current density
- $x$ is cable distance


## Mathematical Foundations

### Hodgkin-Huxley Model

Alan Hodgkin and Andrew Huxley introduced a mathematical framework for modeling ionic currents of excitable membranes that has become standard. It describes the ionic mechanisms underlying action potential generation. The Hodgkin-Huxley model is a system of four ordinary differential equations (ODEs):

#### Membrane Current Equation

$$ C \frac{dV}{dt} = I_{app} - I_{Na_V} - I_{K_V} - I_L$$

Where:
- $C$ is Membrane capacitance
- $I_{app}$ is applied current
- $I_{Na_V}$ is sodium current
- $I_{K_V}$ is potassium current  
- $I_L$ is passive leak current

#### Ionic Currents

**Sodium Current:**

$$ I_{Na_V} = g_{Na} m^3 h (V - E_{Na}) $$

**Potassium Current:**

$$ I_{K_V} = g_K n^4 (V - E_K) $$

**Leak Current:**

$$ I_L = g_L (V - E_L) $$

Where:
- $g_{Na}$, $g_K$, $g_L$: Maximum conductances
- $E_{Na}$, $E_K$, $E_L$: Reversal potentials
- $m$, $h$, $n$: Gating variables ($0 \le m$, $h$, $n \le 1$)

#### Gating Variable Dynamics
The transition of each subunit gate can be described by a kinetic scheme, in which the gating transition *closed → open* occurs at a voltage-dependent rate $\alpha_n(V)$, and the reverse transition *open → closed*, occurs at a voltage dependent rate $\beta_n(V)$.

So the gating variables follow first-order kinetics:

$$ \frac{dn}{dt} = \alpha_n(V) (1 - n) - \beta_n(V) n $$


Where $n$ *= (m, h, n)*, $\alpha_n$, and $\beta_n$ are voltage-dependent rate constants:

**Sodium activation (m):**

$$ \alpha_m = \frac{0.1 (V + 40)}{1 - exp(-0.1 (V + 40))} $$

$$ \beta_m = 4 exp(-0.0556 (V + 65)) $$

**Sodium inactivation (h):**

$$ \alpha_h = 0.07 exp(-0.05 (V + 65)) $$

$$ \beta_h = \frac{1}{1 + exp(-0.1 (V + 35))} $$

**Potassium activation (n):**

$$ \alpha_n = \frac{0.01 (V + 55)}{1 - exp(-0.1 (V + 55))} $$

$$ \beta_n = 0.125 exp(-0.0125 (V + 65)) $$

#### Hodgkin-Huxley Dynamics
A graphical visualization of the temporal evolution of the dynamic variables of the Hodgkin-HUxley model during a single action potential is shown below:

<img src="imgs/hodgkin-huxley dynamics.png" alt="Hodgkin-Huxley Dynamics" width="280">

The upper-most trace is the membrane potential, the second is the membrane current produced by the sum of the Hodgkin-Huxley $K^+$ and $Na^+$ conductances. Subsequent traces show the temporal evolution of $m$, $h$, and $n$. Current injection was initiated at t=5ms. Dayan & Abbott (2001)


## Architecture

### Core Components

#### 1. Compartment Class
- Implements Hodgkin-Huxley dynamics
- Manages ion channels and gating variables
- Handles synaptic inputs

#### 2. Neuron Class
- Manages multiple compartments (soma, axon, dendrites)
- Handles connections between the neurons' compartments
- Provides interface for neuron operations

#### 3. Network Class
- Manages multiple neurons
- Handles synaptic connections between neurons
- Coordinates network simulation

#### 4. Parameter Classes
- `CompartmentParameters`: Morphological and electrical properties
- `NeuronParameters`: Neuron settings
- `SynapticParameters`: Synapse properties

### Compartment Types

#### Soma
- High sodium and potassium conductances
- Primary spike initiation site
- Standard Hodgkin-Huxley parameters

#### Axon
- Enhanced sodium conductance for reliable propagation
- Lower capacitance for faster dynamics
- Long length for signal transmission

#### Dendrites
- Additional ion channels
- Calcium dynamics for plasticity
- Variable lengths and diameters

### Class Diagram
<img src="imgs/class diagram.png" alt="Class Diagram" width="600">

### Structure

- `src/` - Main implementation
  - `neuron.py` - Multi-compartment neuron class
  - `compartment.py` - Individual compartment with Hodgkin-Huxley dynamics
  - `network.py` - Network of connected neurons
  - `parameters.py` - Model parameters and config
  - `demo.py` - Example simulation and visualization

## Installation and Usage

### Requirements
- Python 3.7+
- NumPy
- Matplotlib
- Seaborn
- Networkx

```bash
pip install numpy matplotlib seaborn networkx
```

### Quick Start

```bash
cd src/
python demo.py
```

### Basic Usage

```python
# Create a single neuron
neuron = Neuron(neuron_id=0)
soma_id, axon_id, dendrite_ids = neuron.create_morphology(num_dendrites=2)

# Add external current
neuron.update({soma_id: 0.1})  # 0.1 nA current injection

# Create a simple network with 3 neurons
network = NeuronalNetwork(num_neurons=3)
network.connect_neurons(0, 1, weight=1.5, delay=1.0)

network.simulate(duration=100.0)  # 100 ms simulation
```

## Visualization

The project includes comprehensive visualization tools:

### Network Architecture
- Graph representation of neuron connections
- Layer-based organization
- Connection weight visualization

### Raster Plots
- Spike timing across neurons
- Color-coded by neuron type
- Time-resolved activity patterns

### Firing Rate Analysis
- Per-neuron firing rates
- Statistical summaries
- Activity distribution

### Parameter Analysis
- Neuron parameter comparisons
- Threshold vs. refractory period plots
- Morphological property distributions

### Connection Matrix
- Weight matrices
- Connection density analysis
- Network topology visualization


## GUI Wrapper

A graphical user interface (GUI) allows easier configuration and simulation.

<img src="imgs/gui/application screenshot.png" alt="Screenshot of the running gui application" width="650">


## References

1. Dayan & Abbott (2001). *Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems*. MIT Press.

2. Smith (2019). *Cellular Biophysics and Modeling: A Primer on the Computational Biology of Escitable Cells*. Cambridge University Press.

3. Churchland & Sejnowski (1997) *Grundlagen zur Neuroinformatik und Neurobiologie*. Vieweg.


*A song on the equations to get into the right mood: [The Spark Within the Veil](https://suno.com/s/8Sy206NdDnw5yzba)*
