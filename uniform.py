from numpy import array, prod, sign
from numpy.random import uniform
from math import factorial
from itertools import product

def ab_convert(a: array, b: array) -> list:
    '''Transforms parameters from U(a,b) -> U(c-a, c+a)'''
    c = (a + b) / 2
    return [b-c, c]

def uniform_sum(x: array, a: array, c:array) -> array:
    '''
    PDF of the sum of n uniform distributions on the intervals [c-a, c+a].
    Bradley, D. M. & Gupta, R. C. (2002)  On the Distribution of the Sum of n Non-Identically Distributed Uniform Random Variables. Annals of the Institute of Statistical Mathematics 54.
    See Equation 2.3

    '''
    assert len(a) == len(c)
    n = len(a)

    ## Vector of e combinations (2**n vector)
    epsilon = product([1,-1],repeat= n)

    ## Numerator component
    numerator = 0
    for e in epsilon:
        y = (x+sum(e*a-c))
        numerator += (y**(n-1) * sign(y) * prod(e))

    ## Denominator component
    denominator = factorial(n-1) * 2**(n+1) * prod(a)
    
    return numerator / denominator

def uniform_sum_sim(a:array, b:array, n:int = 10**6) -> array:
    '''
    Runs a simulations for the sum of n independent U(a,b) random variables
    '''   
    assert len(a) == len(b)
    
    simulation = 0 
    for i in range(len(a)):
        simulation += uniform(low=a[i], high=b[i], size=n)

    return simulation
