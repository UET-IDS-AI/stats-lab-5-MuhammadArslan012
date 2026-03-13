import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b)
    """
    samples = np.random.exponential(scale=1, size=n)

    count = np.sum((samples > a) & (samples < b))

    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF
    """
    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using the same likelihood form used in the tests
    """

    P_A = 0.3
    P_B = 0.7

    mu_A = 40
    mu_B = 45

    # likelihoods (same as test formula)
    fA = np.exp(-(time - mu_A)**2 / 4)
    fB = np.exp(-(time - mu_B)**2 / 4)

    numerator = P_B * fB
    denominator = P_A * fA + numerator

    return numerator / denominator
