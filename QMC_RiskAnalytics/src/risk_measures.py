from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import WeightedAdder, IntegerComparator
from qiskit.primitives import StatevectorSampler

import numpy as np

from qmc_distributions import BinomialTreeModel
from utils import interpolate_inverse_cdf

class ValueAtRisk:

    def __init__(self, alpha, distribution_model):

        self.alpha = alpha
        self.D = distribution_model

        self.m = self.D.m
        self.u = self.D.u
        self.d = self.D.d

        self.num_anc = int(np.ceil(np.log2(self.m)))

        self.perc_changes = self._compute_losses()

    def _compute_losses(self):
        # TODO: Move to QMC Distribution
        decimal_changes = np.array([np.power(self.u, m) * np.power(self.d, (self.m - m)) - 1.0 for m in range(self.m + 1)])
        perc_changes = (decimal_changes * 100).astype(int)

        return perc_changes

    def risk_measure_circuit(self, threshold):
        d = self.D.distribution_circuit()
        adder = WeightedAdder(num_state_qubits=d.num_qubits)
        comparator = IntegerComparator(num_state_qubits=adder.num_sum_qubits, value=threshold, geq=False)

        rf = QuantumRegister(d.num_qubits, 'rf')
        add = QuantumRegister(adder.num_sum_qubits, 'add')
        car = QuantumRegister(adder.num_carry_qubits, 'car')
        ctl = QuantumRegister(adder.num_control_qubits, 'ctl')
        cmp = QuantumRegister(comparator.num_ancillas, 'cmp')
        rm = QuantumRegister(1, 'rm')

        qc = QuantumCircuit(rf, add, car, ctl, cmp, rm, name='risk_measure_circuit')
        qc.append(d.to_gate(), rf)
        qc.append(adder.to_gate(), rf[:] + add[:] + car[:] + ctl[:])
        qc.append(comparator.to_gate(), add[:] + rm[:] + cmp[:])
        qc.append(adder.to_gate().inverse(), rf[:] + add[:] + car[:] + ctl[:])

        return qc


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import seaborn as sns
    from qiskit.visualization import plot_histogram

    plot = True
    threshold = 3

    btm = BinomialTreeModel(mu=0.08, sigma=0.2, T=1, num_steps=6)
    var = ValueAtRisk(0.05, btm)
    qc = var.risk_measure_circuit(threshold=threshold)
    if plot:
        qc.draw(output="mpl")
        plt.show()

    qc.measure_all()
    sampler = StatevectorSampler()
    result = sampler.run([qc]).result()
    data = result[0].data
    raw_counts = data.meas.get_counts()
    probs = {k: v / data.meas.num_shots for k, v in raw_counts.items()}

    cdf_prob = 0
    for i, prob in probs.items():
        if prob > 1e-6 and i[0][0] == "1":
            cdf_prob += prob

    reduced_raw = {}
    sums_meas = {}
    comps = 0
    for k, v in raw_counts.items():
        invk = k[::-1]
        new_k = invk[:6]
        pop_cnt = int(new_k, 2).bit_count()
        sum_bits = invk[6:9][::-1]
        sum_val = int(sum_bits, 2)
        comp_bit = invk[-1]
        ground_truth = int(sum_val < threshold)
        if new_k in reduced_raw:
            reduced_raw[new_k] += v
        else:
            reduced_raw[new_k] = v

        if sum_bits in sums_meas:
            sums_meas[sum_bits] += (v / data.meas.num_shots)
        else:
            sums_meas[sum_bits] = (v / data.meas.num_shots)

        if comp_bit == '1':
            comps += (v / data.meas.num_shots)

    cnt = BinomialTreeModel.convert_measurement(reduced_raw, ket_labels=True)

    probs = BinomialTreeModel.convert_measurement(reduced_raw, ket_labels=False)
    losses = var.perc_changes / 100
    prob_mass = np.array([[k, losses[k], probs[k], 0] for k in sorted(probs.keys())])
    prob_mass[:, 3] = np.cumsum(prob_mass[:, 2])

    print(prob_mass)
    print(f"Estimated CDF at u={threshold}: {cdf_prob}")

    alpha = 0.05
    tresholds = prob_mass[:, 3] >= alpha
    tresholds_idx = tresholds.tolist().index(True)
    print(f"{tresholds_idx=}")

    xvals = np.array([losses[tresholds_idx - 1], losses[tresholds_idx]])
    Fvals = np.array([prob_mass[tresholds_idx - 1, 3], prob_mass[tresholds_idx, 3]])

    iVar = interpolate_inverse_cdf(var.alpha, xvals, Fvals)
    print(f"Interpolated {alpha}-Value at risk: {iVar}")

    if plot:
        sns.barplot(data=cnt)
        plt.show()

        # sns.barplot(data=sums_meas)
        # plt.show()

        plt.plot(prob_mass[:, 1], prob_mass[:, 2], 'b-o')
        plt.fill_between(prob_mass[:tresholds_idx + 1, 1], prob_mass[:tresholds_idx + 1, 2], 0, alpha=0.2, color='r')
        plt.axvline(iVar, c='r')
        plt.title("PMF")
        plt.show()

        plt.plot(prob_mass[:, 1], prob_mass[:, 3], 'b-o')
        plt.axvline(iVar, c='r')
        plt.axhline(alpha, c='g')
        plt.title("CDF")
        plt.show()





