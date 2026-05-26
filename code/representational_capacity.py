"""
Closed-form representational capacity formula from the paper.

Implements the fully-parameterized adjusted relationship
    eps = sqrt(C * ln(k^a / d)^b / d^c)
along with its inverse `max_k(d, eps)` (the representational capacity given
d_model and accepted deviation eps). The constants C, a, b, c below are the
fit reported in Section "An Adjusted Relationship for Optimized Vectors";
they are produced by vector_packing_graphs.py from vector_packing/optimized_packing.json.

Re-running vector_packing_graphs.py may yield slightly different constants; if so,
update them here to keep `max_k` consistent with the figures.
"""

import numpy as np
from scipy.optimize import minimize_scalar

C = 0.45809143
a = 1.0668005
b = 1.44738666
c = 0.97197234


def bounds(k, n): return np.sqrt(C * np.log(k ** a / n) ** b / n ** c)


def min_n(k, bounds):
    target = bounds**2 / C

    def equation(n):
        if n <= 0 or n >= k**a:
            return 1e6  # Outside domain
        try:
            log_term = np.log(k**a / n)
            if log_term <= 0:
                return 1e6
            val = (log_term**b) * n**(-c)
            return (val - target)**2  # Minimize squared error
        except:
            return 1e6

    result = minimize_scalar(equation, bounds=(1, k), method='bounded')
    if result.success:
        return np.ceil(result.x)
    else:
        return "Root finding did not converge."


def max_k(n, bounds): return np.ceil((
    n * np.exp((bounds ** 2 * n ** c / C) ** (1 / b))) ** (1 / a))
