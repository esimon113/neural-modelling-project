
import numpy as np

from parameters import CompartmentParameters, CompartmentType


class CompartmentState:
    """
    Parameters:
        `V`:  Membrane potential (mV)
        `m`:  Na+ activation
        `h`:  Na+ inactivation
        `n`:  K+ activation
        `Ca`: Calcium concentration (μM)
        `t`:  Time (ms)
        `I_axial_in`:  Incoming axial current (nA)
        `I_axial_out`: Outgoing axial current (nA)
    """

    V: float = -65.0
    m: float = 0.0
    h: float = 0.0
    n: float = 0.0
    Ca: float = 0.0
    t: float = 0.0

    # Axial currents from or to neihgbors
    I_axial_in: float = 0.0
    I_axial_out: float = 0.0

    # Refractory period tracking
    last_spike_time: float = -1000.0  # Time of last spike (ms)


class Compartment:
    def __init__(self, comp_id: int, comp_type: CompartmentType, params: CompartmentParameters, dt: float = 0.01):
        """
        Args:
            `comp_id`: Unique identifier for this compartment
            `comp_type`: Type of compartment (soma, axon, dendrite)
            `params`: Compartment parameters
            `dt`: Time step (ms)
        """

        self.id = comp_id
        self.type = comp_type
        self.params = params
        self.dt = dt

        self.state = CompartmentState()
        self._initialize_gating_variables()

        self.V_history: list[float] = []
        self.spike_times: list[float] = []
        self.last_V: float = self.state.V

        self.synapses: list['Synapse'] = []

    def _initialize_gating_variables(self):
        V = self.state.V
        self.state.m = self._alpha_m(V) / (self._alpha_m(V) + self._beta_m(V))
        self.state.h = self._alpha_h(V) / (self._alpha_h(V) + self._beta_h(V))
        self.state.n = self._alpha_n(V) / (self._alpha_n(V) + self._beta_n(V))
        self.state.Ca = self.params.Ca_0

    def _alpha_n(self, V: float) -> float:
        x = -0.1 * (V + 55)

        if abs(x) < 1e-6:  # Avoid division by zero
            return 0.01 * (V + 55) / 0.1

        return 0.01 * (V + 55) / (1 - np.exp(np.clip(x, -50, 50)))

    def _beta_n(self, V: float) -> float: return 0.125 * \
        np.exp(np.clip(-0.0125 * (V + 65), -50, 50))
    """
    The activation and inactivation variables `m` and `h` are distinguished be having opposite voltage dependences.
    Depolarization causes `m` to increase and `h` to decrease, and hyperpolarization decreases `m` while increasing `h`.
    (Dayan & Abbott, 2001, p.172)
    """

    def _alpha_m(self, V: float) -> float:
        x = -0.1 * (V + 40)

        if abs(x) < 1e-6:  # Avoid division by zero
            return 0.1 * (V + 40) / 0.1

        return 0.1 * (V + 40) / (1 - np.exp(np.clip(x, -50, 50)))

    def _beta_m(self, V: float) -> float: return 4.0 * \
        np.exp(np.clip(-0.0556 * (V + 65), -50, 50))

    def _alpha_h(self, V: float) -> float: return 0.07 * \
        np.exp(np.clip(-0.05 * (V + 65), -50, 50))

    def _beta_h(self, V: float) -> float: return 1 / \
        (1 + np.exp(np.clip(-0.1 * (V + 35), -50, 50)))

    # Dendrite channel rate constants
    def _alpha_Ca(self, V: float) -> float:
        x = -0.1 * (V + 20)

        if abs(x) < 1e-6:  # Avoid division by zero
            return 0.1 * (V + 20) / 0.1

        return 0.1 * (V + 20) / (1 - np.exp(np.clip(x, -50, 50)))

    def _beta_Ca(self, V: float) -> float: return 0.1 * \
        np.exp(np.clip(-0.05 * (V + 20), -50, 50))

    def _alpha_K_Ca(self, Ca: float) -> float: return 0.1 * Ca / (Ca + 0.1)
    def _beta_K_Ca(self, Ca: float) -> float: return 0.1

    def _alpha_h_channel(self, V: float) -> float: return 0.1 * \
        np.exp(np.clip(0.05 * (V + 80), -50, 50))

    def _beta_h_channel(self, V: float) -> float: return 0.1 / \
        (1 + np.exp(np.clip(0.05 * (V + 80), -50, 50)))

    def _alpha_Kv(self, V: float) -> float:
        x = -0.1 * (V + 30)

        if abs(x) < 1e-6:  # Avoid division by zero
            return 0.1 * (V + 30) / 0.1

        return 0.1 * (V + 30) / (1 - np.exp(np.clip(x, -50, 50)))

    def _beta_Kv(self, V: float) -> float: return 0.1 * \
        np.exp(np.clip(-0.05 * (V + 30), -50, 50))

    def update(self, I_ext: float = 0.0) -> bool:
        """
        Update state, returns true if spiked

        Args:
            `I_ext`: External current injection (nA)
        """

        self.V_history.append(self.state.V)

        I_Na, I_K, I_L = self._calculate_hodgkin_huxley_currents()
        I_Ca, I_K_Ca, I_h, I_Kv = self._calculate_dendritic_currents()
        I_syn = sum(synapse.get_current(self.state.V)
                    for synapse in self.synapses)
        I_membrane = I_Na + I_K + I_L + I_Ca + I_K_Ca + I_h + I_Kv + I_syn
        I_axial = self.state.I_axial_in - self.state.I_axial_out

        dV_dt = (I_ext + I_axial - I_membrane) / \
            (self.params.C_m * self.params.area * 1e-8)
        self.state.V += dV_dt * self.dt

        # clip voltage to prevent overflow
        self.state.V = np.clip(self.state.V, -200.0, 200.0)

        self._update_gating_variables()
        self._update_calcium_concentration(I_Ca)

        spiked = self._has_spiked()
        self.state.t += self.dt

        return spiked

    def _calculate_hodgkin_huxley_currents(self) -> tuple[float, float, float]:
        V = self.state.V

        m = np.clip(self.state.m, 0.0, 1.0)
        h = np.clip(self.state.h, 0.0, 1.0)
        n = np.clip(self.state.n, 0.0, 1.0)

        g_Na = self.params.g_Na * m**3 * h
        g_K = self.params.g_K * n**4
        g_L = self.params.g_L

        g_Na = np.clip(g_Na, 0.0, 1000.0)  # Max 1000 mS/cm²
        g_K = np.clip(g_K, 0.0, 1000.0)
        g_L = np.clip(g_L, 0.0, 100.0)

        I_Na = g_Na * (V - self.params.E_Na)
        I_K = g_K * (V - self.params.E_K)
        I_L = g_L * (V - self.params.E_L)

        return I_Na, I_K, I_L

    def _calculate_dendritic_currents(self) -> tuple[float, float, float, float]:
        V = self.state.V
        I_Ca = I_K_Ca = I_h = I_Kv = 0.0

        # Calcium
        if self.params.g_Ca > 0:
            alpha_Ca = self._alpha_Ca(V)
            beta_Ca = self._beta_Ca(V)

            if alpha_Ca + beta_Ca > 0:
                g_Ca = self.params.g_Ca * alpha_Ca / (alpha_Ca + beta_Ca)
                g_Ca = np.clip(g_Ca, 0.0, 1000.0)
                I_Ca = g_Ca * (V - self.params.E_Ca)

        # Calcium-activated ptoassium
        if self.params.g_K_Ca > 0 and self.state.Ca > 0:
            alpha_K_Ca = self._alpha_K_Ca(self.state.Ca)
            beta_K_Ca = self._beta_K_Ca(self.state.Ca)

            if alpha_K_Ca + beta_K_Ca > 0:
                g_K_Ca = self.params.g_K_Ca * \
                    alpha_K_Ca / (alpha_K_Ca + beta_K_Ca)
                g_K_Ca = np.clip(g_K_Ca, 0.0, 1000.0)
                I_K_Ca = g_K_Ca * (V - self.params.E_K)

        # HCN
        if self.params.g_h > 0:
            alpha_h = self._alpha_h_channel(V)
            beta_h = self._beta_h_channel(V)

            if alpha_h + beta_h > 0:
                g_h = self.params.g_h * alpha_h / (alpha_h + beta_h)
                g_h = np.clip(g_h, 0.0, 1000.0)
                I_h = g_h * (V - self.params.E_L)

        # Voltage-gated potassium
        if self.params.g_Kv > 0:
            alpha_Kv = self._alpha_Kv(V)
            beta_Kv = self._beta_Kv(V)

            if alpha_Kv + beta_Kv > 0:
                g_Kv = self.params.g_Kv * alpha_Kv / (alpha_Kv + beta_Kv)
                g_Kv = np.clip(g_Kv, 0.0, 1000.0)
                I_Kv = g_Kv * (V - self.params.E_K)

        return I_Ca, I_K_Ca, I_h, I_Kv

    def _update_gating_variables(self):
        V = self.state.V

        # Update m, h, n
        self.state.m += self.dt * \
            (self._alpha_m(V) * (1 - self.state.m) - self._beta_m(V) * self.state.m)
        self.state.h += self.dt * \
            (self._alpha_h(V) * (1 - self.state.h) - self._beta_h(V) * self.state.h)
        self.state.n += self.dt * \
            (self._alpha_n(V) * (1 - self.state.n) - self._beta_n(V) * self.state.n)

        self.state.m = np.clip(self.state.m, 0.0, 1.0)
        self.state.h = np.clip(self.state.h, 0.0, 1.0)
        self.state.n = np.clip(self.state.n, 0.0, 1.0)

    def _update_calcium_concentration(self, I_Ca: float):
        if self.params.g_Ca > 0 and self.params.area > 0:
            ca_influx = I_Ca / (2 * 96485 * self.params.area * 1e-8)
            ca_decay = self.state.Ca / self.params.tau_Ca
            self.state.Ca = max(0, self.state.Ca +
                                self.dt * (ca_influx - ca_decay))

    def _has_spiked(self) -> bool:
        # Check if refractory period is over
        time_since_last_spike = self.state.t - self.state.last_spike_time
        in_refractory = time_since_last_spike < self.params.refractory_period

        spike_threshold = self.params.spike_threshold if hasattr(
            self.params, 'spike_threshold') else 0.0

        # threshold crossed from below AND not during refractory period
        spiked = (self.last_V < spike_threshold and self.state.V >=
                  spike_threshold) and not in_refractory

        if spiked:
            self.spike_times.append(self.state.t)
            self.state.last_spike_time = self.state.t

        self.last_V = self.state.V

        return spiked

    def reset_to_init_state(self):
        self.state = CompartmentState()
        self._initialize_gating_variables()
        self.V_history = []
        self.spike_times = []
        self.last_V = self.state.V

        for synapse in self.synapses:
            synapse.reset_to_init_state()

    def add_synapse(self, synapse: 'Synapse'): self.synapses.append(synapse)

    def get_voltage_history(
        self) -> np.ndarray: return np.array(self.V_history)

    def get_spike_times(self) -> list[float]: return self.spike_times.copy()

    def has_spiked(self) -> bool: return (len(self.spike_times)
                                          > 0 and self.spike_times[-1] == self.state.t - self.dt)


class Synapse:
    """
    COntains implementation of connection between eneuron compartments. 
    Implements alpha function synaptic dynamics with configurable rise and decay time constants.
    """

    def __init__(self, tau_rise: float = 0.5, tau_decay: float = 5.0, amplitude: float = 1.0, reversal_potential: float = 0.0, dt: float = 0.01):
        """
        Initializes synapse

        Args:
            `tau_rise`: Rise time constant (ms)
            `tau_decay`: Decay time constant (ms)
            `amplitude`: Peak conductance (nS)
            `reversal_potential`: Reversal potential (mV)
            `dt`: Time step (ms)
        """

        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.amplitude = amplitude
        self.E_rev = reversal_potential
        self.dt = dt

        self.g = 0.0  # Current conductance
        self.t = 0.0  # Time

    def get_current(self, V_membrane: float) -> float:
        """
        Returns Synaptic current (nA).

        Args:
            `V_membrane`: Membrane potential (mV)
        """

        self.g *= np.exp(-self.dt / self.tau_decay)  # decay
        return self.g * (V_membrane - self.E_rev)

    def add_spike(self, spike_time: float): self.g += self.amplitude

    def update_state_time(self, dt: float): self.t += dt

    def reset_to_init_state(self):
        self.g = 0.0
        self.t = 0.0
