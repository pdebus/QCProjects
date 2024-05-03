from qiskit import QuantumCircuit, QuantumRegister

import numpy as np


class BinomialTreeModel:
    def __init__(self, mu, sigma, T, num_steps):
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.m = num_steps
        self.dt = T / num_steps

        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u

        eps = 1e-15
        self.q = (self.u * np.exp(self.mu * self.dt) - 1 + eps) / (self.u ** 2 - 1 + eps)

    def distribution_circuit(self):
        rf = QuantumRegister(self.m, 'rf')
        qc = QuantumCircuit(rf, name="BinomialTree")

        theta_u = 2 * np.arcsin(np.sqrt(self.q))

        for i in range(self.m):
            qc.ry(theta_u, rf[i])

        return qc

    def distribution_gate(self):
        qc = self.distribution_circuit()
        return qc.to_gate()

    @staticmethod
    def convert_measurement(raw_counts, normalize=True, ket_labels=False):
        counts = {}
        for k, v in raw_counts.items():
            pop_cnt = int(k, 2).bit_count()
            if pop_cnt not in counts:
                counts[pop_cnt] = v
            else:
                counts[pop_cnt] += v

        total_counts = np.sum(list(counts.values()))
        if normalize:
            counts = {k: v / total_counts for k, v in counts.items()}

        if ket_labels:
            counts = {r"$|\# u=" + str(k) + r"\rangle$": counts[k] for k in sorted(counts.keys())}

        return counts

    def plot_histogram(self, shots=2048):
        from qiskit.primitives import StatevectorSampler

        from qiskit.visualization import plot_histogram
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")

        qc = self.distribution_circuit()
        qc.measure_all()

        sampler = StatevectorSampler()
        result = sampler.run([qc], shots=shots).result()
        data = result[0].data
        raw_counts = data.meas.get_counts()

        #plot_histogram(raw_counts)
        #plt.show()

        counts = BinomialTreeModel.convert_measurement(raw_counts, ket_labels=True)

        labels = []
        for m in range(self.m + 1):
            uterm = "u^{" + str(m) + "}" if m > 0 else ""
            dterm = "d^{" + str(self.m - m) + "}" if (self.m - m) > 0 else ""
            label = f"$S_0{uterm}{dterm}$"
            labels.append(label)

        ax = sns.barplot(data=counts)
        ax.bar_label(container=ax.containers[0], labels=labels, fontsize=10)
        ax.set_title(f"Distribution of $S_T(\mu={self.mu:.2f}, \sigma={self.sigma:.2f})$ at T={self.T}, "
                     f"q={self.q:.2f}, u={self.u:.2f}")
        ax.set_xlabel(r"$S_T$")
        ax.set_ylabel(r"$\mathbb{P}\left(|\psi_{rf}\rangle=|\# u=k\rangle \right)$")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    btm = BinomialTreeModel(mu=0.08, sigma=0.2, T=1, num_steps=6)
    btm.plot_histogram()
