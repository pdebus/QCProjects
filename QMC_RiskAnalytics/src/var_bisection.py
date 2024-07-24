from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import WeightedAdder, IntegerComparator
from qiskit.primitives import StatevectorSampler

import numpy as np

from qmc_distributions import BinomialTreeModel
from risk_measures import ValueAtRisk
from utils import interpolate_inverse_cdf


def var_bisection_search(risk_measure):
    def evaluator(u):

        qc = risk_measure.risk_measure_circuit(threshold=u)
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

        return cdf_prob - risk_measure.alpha

    #for u in range(0, risk_measure.m + 1):
    #    Fu = evaluator(u) + risk_measure.alpha
    #    print(f"F({u})={Fu} <= {risk_measure.alpha}: {Fu <= risk_measure.alpha} ")

    left = 0
    right = risk_measure.m // 2

    pl = evaluator(left)
    pr = evaluator(right)

    #print(f"left: {left}, right: {right}")
    #print(f"pl: {pl}, pr: {pr}")

    if pl * pr > 0:
        raise ValueError

    while right - left > 1:
        c = (right + left) // 2
        pc = evaluator(c)
        #print(f"\nleft: {left}, right: {right}")
        #print(f"pl: {pl}, pr: {pr}")
        #print(f"c={c}, pc={pc}")
        if pl * pc < 0:
            right = c
            pr = pc
        else:
            left = c
            pl = pc

    return [left, right], [pl + risk_measure.alpha, pr + risk_measure.alpha]


if __name__ == "__main__":

    btm = BinomialTreeModel(mu=0.08, sigma=0.2, T=1, num_steps=6)
    var = ValueAtRisk(0.05, btm)

    results = var_bisection_search(var)
    u = results[0][0]
    Fu = results[1][0]
    losses = var.perc_changes / 100
    print(f"u={u} => F(u)={Fu}\nmin u s.t. F(u)>={var.alpha}")
    print("Value at risk for u=2:", losses[u])

    uvals = np.array(results[0])
    xvals = np.array([losses[i] for i in results[0]])
    Fvals = np.array(results[1])

    iu = interpolate_inverse_cdf(var.alpha, uvals, Fvals)
    iVar = interpolate_inverse_cdf(var.alpha, xvals, Fvals)

    print(f"Interpolated {var.alpha}-Value at risk: {iVar}\n(u={iu})")


