from numpy.core.shape_base import vstack
from lib import Generator, assesse_components, assesse_combination, mises

import matplotlib.pyplot as plt
import numpy as np


# initial data 
#       A -->
#   B
#   |
#   V

theoretical_p = np.array([
    [0.1, 0.1, 0.1],
    [0.05, 0.2, 0.05],
    [0.2, 0.1, 0.1]
])

theoretical_p_A = np.array([0.35, 0.4, 0.25])
theoretical_p_B = np.array([0.3, 0.3, 0.4])

A = [1, 2, 3]
B = [1, 2, 3]

A_sample = []
B_sample = []
A_values = np.array([1, 2, 3])
B_values = np.array([1, 2, 3])
N = 100000

generator = Generator(theoretical_p, A, B)
empirical_p = np.zeros((
    len(theoretical_p), 
    len(theoretical_p[0])
))

for _ in range(N):
    Ai, Bi = generator.generate()
    empirical_p[Bi - 1][Ai - 1] += 1
    A_sample.append(Ai)
    B_sample.append(Bi)

# Empirical probabilites
empirical_p = (1 / N) * empirical_p
empirical_p_A = np.sum(empirical_p, axis=0)
empirical_p_B = np.sum(empirical_p, axis=1)
print('Empirical probabilites: \n', empirical_p)

def theoretical_A_CDF(a):
    if a <= 1:
        return 0

    if 1 < a and a <= 2:
        return 0.35

    if 2 < a and a <= 3:
        return 0.75

    if a > 3:
        return 1

def theoretical_B_CDF(b):
    if b <= 1:
        return 0

    if 1 < b and b <= 2:
        return 0.3

    if 2 < b and b <= 3:
        return 0.6

    if b > 3:
        return 1

def empirical_A_CDF(a):
    if a <= 1:
        return 0

    if 1 < a and a <= 2:
        return empirical_p_A[0]

    if 2 < a and a <= 3:
        return empirical_p_A[0] + empirical_p_A[1] 

    if a > 3:
        return 1

def empirical_B_CDF(b):
    if b <= 1:
        return 0

    if 1 < b and b <= 2:
        return empirical_p_B[0]

    if 2 < b and b <= 3:
        return empirical_p_B[0] + empirical_p_B[1] 

    if b > 3:
        return 1


# Histograms
plt.title('A component histogram')
plt.hist(A_sample)
plt.show()

plt.title('B component histogram')
plt.hist(B_sample)
plt.show()

plt.title('AB histogram')
plt.hist2d(A_sample, B_sample)
plt.show()

# statistics assessment
assesse_components(A_sample, A_values, theoretical_p_A, 'A')
print()

assesse_components(B_sample, B_values, theoretical_p_B, 'B')
print()

assesse_combination(A_sample, B_sample, A_values, B_values, theoretical_p)
print()

print('Mises criteria')
print('A component')
mises(len(A), theoretical_A_CDF, empirical_A_CDF)
print()
print('B component')
mises(len(B), theoretical_A_CDF, empirical_A_CDF)