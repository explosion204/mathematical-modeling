import numpy as np
from statistics import mean
from numpy import random
from numpy.lib.scimath import sqrt
from scipy.optimize import fsolve
from scipy.stats import chi2
from scipy.special import erf
from math import exp

class Generator:
    def __init__(self, p, A, B):
        self.p = p
        self.A = A
        self.B = B

    def generate(self): 
        x = np.random.uniform()
        pi = iter(self.p.flatten())
        pi_prev = 0
        pi_current = next(pi)
        k = 0
                
        while True:    
            if pi_prev < x and x <= pi_current:
                break

            k += 1
            try:
                pi_prev = pi_current
                delta = next(pi)
                pi_current = pi_current + delta
            except StopIteration:
                break

        i = k // len(self.p)
        j = k % len(self.p[0])

        return self.A[j], self.B[i]

# level of importance = 0.01
NORM_QUANTILE = 2.5758293

def chi_square_quantile(N, a):
    return fsolve(lambda x: chi2.cdf(x, N - 1) - a, N)[0], fsolve(lambda x: chi2.cdf(x, N - 1) - (1 - a), N)[0]

def laplace_quantile(y):
    return fsolve(lambda x: erf(x) - y, 0)[0]

def assesse_components(sample, values, p, component_name):
    empirical_mean = mean(sample)
    theoretical_mean = np.sum([value * pi for value, pi in zip(values, p)])
    empirical_variance = sum([(x - empirical_mean) ** 2 for x in sample]) / (len(sample) - 1)

    theoretical_mean_of_square = np.sum([(value ** 2) * pi for value, pi in zip(values, p)])
    theoretical_variance = theoretical_mean_of_square - theoretical_mean ** 2

    mean_delta = NORM_QUANTILE * sqrt(empirical_variance) / sqrt(len(sample))

    # a / 2 & 1 - a / 2
    chi_square_quantile_upper, chi_square_quantile_lower = chi_square_quantile(len(sample), 0.005)
    variance_lower = (len(sample) - 1) * empirical_variance / chi_square_quantile_lower
    variance_upper = (len(sample) - 1) * empirical_variance / chi_square_quantile_upper

    print(f'Empirical mean of component {component_name}: {empirical_mean}')
    print(f'Theoretical mean of component {component_name}: {theoretical_mean}')
    print(f'Confidence interval of mean: ({empirical_mean - mean_delta}, {empirical_mean + mean_delta})')

    print(f'Empirical variance of component {component_name}: {empirical_variance}')
    print(f'Theoretical variance of component {component_name}: {theoretical_variance}')
    print(f'Confidence interval of variance: ({variance_lower}, {variance_upper})')

def assesse_combination(A_sample, B_sample, A_values, B_values, p):
    empirical_mean_A = mean(A_sample)
    empirical_mean_B = mean(B_sample)
    empirical_variance_A = sum([(x - empirical_mean_A) ** 2 for x in A_sample]) / (len(A_sample) - 1)
    empirical_variance_B = sum([(x - empirical_mean_B) ** 2 for x in B_sample]) / (len(B_sample) - 1)
    theoretical_mean_A = np.sum([value * pi for value, pi in zip(A_values, p)])
    theoretical_mean_B = np.sum([value * pi for value, pi in zip(B_values, p)])

    theoretical_mean_of_square_A = np.sum([(value ** 2) * pi for value, pi in zip(A_values, p)])
    theoretical_variance_A = theoretical_mean_of_square_A - theoretical_mean_A ** 2

    theoretical_mean_of_square_B = np.sum([(value ** 2) * pi for value, pi in zip(B_values, p)])
    theoretical_variance_B = theoretical_mean_of_square_B - theoretical_mean_B ** 2

    empirical_K = np.sum([(a - empirical_mean_A) * (b - empirical_mean_B) for a, b in zip(A_sample, B_sample)]) / (len(A_sample))
    empirical_R = empirical_K / (sqrt(empirical_variance_A) * sqrt(empirical_variance_B))

    theoretical_K = np.sum([(b - theoretical_mean_A) * (a - theoretical_mean_B) * p[i][j] for i, b in enumerate(B_values) for j, a in enumerate(A_values)])
    theoretical_R = theoretical_K / (sqrt(theoretical_variance_A) * sqrt(theoretical_variance_B))

    z = laplace_quantile(0.005)
    a = 0.5 * np.log((1 + empirical_R) / (1 - empirical_R)) - z / sqrt(len(A_sample) - 3)
    b = 0.5 * np.log((1 + empirical_R) / (1 - empirical_R)) + z / sqrt(len(B_sample) - 3)
    R_lower = (exp(2 * a) - 1) / (exp(2 * a) + 1) 
    R_upper = (exp(2 * b) - 1) / (exp(2 * b) + 1) 

    print(f'Empirical correlational moment: {empirical_K}')
    print(f'Theoretical correlational moment: {theoretical_K}')
    print(f'Empirical correlational coefficient: {empirical_R}')
    print(f'Theoretical correlational coefficient: {theoretical_R}')
    print(f'Confidence interval of correlational coefficient: ({R_lower}, {R_upper})')

def mises(values_num, theoretical_CDF, empirical_CDF):
    observable_value = 1 / (12 * values_num) + np.sum((empirical_CDF(x) - theoretical_CDF(x)) ** 2 for x in range(values_num))
    critical_value = 0.347
    print(f'Critical value (a = 0.01) = {critical_value}')
    print(f'Criteria value: {observable_value}')

    if observable_value < critical_value:
        print('Converging')
    else:
        print('Not converging')