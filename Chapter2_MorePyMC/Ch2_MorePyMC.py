# vim:fileencoding=utf-8

import pymc3 as pm
import numpy as np
from scipy import stats
import theano.tensor as tt


def test1():
    with pm.Model() as model:
        lambda_1 = pm.Exponential("lambda_1", 1.0)
        lambda_2 = pm.Exponential("lambda_2", 1.0)
        tau = pm.DiscreteUniform("tau", lower=0, upper=10)

    with model:
        new_deterministic_variable = lambda_1 + lambda_2
        print((type(new_deterministic_variable), new_deterministic_variable))

    n_data_points = 5  # in CH1 we had ~70 data points
    idx = np.arange(n_data_points)
    with model:
        lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)

def test2():
    def subtract(x, y):
        return x - y # implicitly return a deterministic variable

    with pm.Model() as model:
        stochastic_1 = pm.Uniform("U_1", 0, 1)
        stochastic_2 = pm.Uniform("U_2", 0, 1)

    # The explanation above may be wrong. Since subtract automatically
    # returns a deterministic variable, no need to explicitly create
    # one. If it were to create, pm.Deterministic must inside "with model:"
    det_1 = subtract(stochastic_1, stochastic_2)
    print((type(det_1), det_1))

