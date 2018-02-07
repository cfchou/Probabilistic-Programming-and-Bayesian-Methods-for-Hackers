# vim:fileencoding=utf-8

import pymc3 as pm
import numpy as np
from scipy import stats
import theano.tensor as tt


def task1():
    with pm.Model() as model:
        lambda_1 = pm.Exponential("lambda_1", 1.0)
        lambda_2 = pm.Exponential("lambda_2", 1.0)
        tau = pm.DiscreteUniform("tau", lower=0, upper=10)

        # (<class 'pymc3.model.FreeRV'>, tau)
        # class FreeRV(Factor, TensorVariable):
        print((type(tau), tau))

    with model:
        new_deterministic_variable = lambda_1 + lambda_2
        # (<class 'theano.tensor.var.TensorVariable'>, Elemwise{add,no_inplace}.0)
        print((type(new_deterministic_variable), new_deterministic_variable))

    n_data_points = 5  # in CH1 we had ~70 data points
    idx = np.arange(n_data_points)
    with model:
        lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)
        print(type(lambda_), lambda_)
    print("basic_RVs:", model.basic_RVs)
    print("free_RVs:", model.free_RVs)
    print("observed_RVs:", model.observed_RVs)
    print("unobserved_RVs:", model.unobserved_RVs)

def task2():
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

def task3():
    with pm.Model() as theano_test:
        p1 = pm.Uniform("p", 0, 1)
        p2 = 1 - p1
        p = tt.stack([p1, p2])
        assignment = pm.Categorical("assignment", p)
        # (<class 'pymc3.model.FreeRV'>, assignment)
        print((type(assignment), assignment))

def task4():
    with pm.Model() as model:
        lambda_1 = pm.Exponential("lambda_1", 1.0)
    # use .random() to explicitly generate samples
    samples = [float(lambda_1.random()) for i in range(20000)]
    print(samples[:10])

def task5():
    data = np.array([10, 5])
    with pm.Model() as model:
        fixed_variable = pm.Poisson("fxd", 1, observed=data)
        print(type(fixed_variable), fixed_variable)
        print("value: ", fixed_variable.tag.test_value)

def task6():
    # generate a number from uniform
    tau = np.random.randint(0, 80)
    print(tau)

    # generate two numbers from exponential
    alpha = 1./20.
    # Here lambdas are concrete values sampled from the same dist.
    # Note previously they are stochastic variables independently generated from two dists.
    lambda_1, lambda_2 = np.random.exponential(scale=1/alpha, size=2)
    print(lambda_1, lambda_2)

    # numpy.random generate one value at a time
    # scipy.stats can generate many by .rvs(size=N, ...)
    # generate many numbers from poisson
    data = np.r_[stats.poisson.rvs(mu=lambda_1, size=tau), stats.poisson.rvs(mu=lambda_2, size = 80 - tau)]

def task7():
    # all use scipy.stats
    def plot_artificial_sms_dataset():
        # generate many from uniform
        tau = stats.randint.rvs(0, 80)
        alpha = 1./20.
        # generate many from exponential
        lambda_1, lambda_2 = stats.expon.rvs(scale=1/alpha, size=2)
        # generate many from
        data = np.r_[stats.poisson.rvs(mu=lambda_1, size=tau), stats.poisson.rvs(mu=lambda_2, size=80 - tau)]
        plt.bar(np.arange(80), data, color="#348ABD")
        plt.bar(tau - 1, data[tau-1], color="r", label="user behaviour changed")
        plt.xlim(0, 80);



if __name__ == "__main__":
    task1()
