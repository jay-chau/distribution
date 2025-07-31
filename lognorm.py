from numpy import array, log, exp

## Descriptive
def mean(mu: array, sigma_sq: array) -> array:
    '''Calculates the mean of n log-normal distributions'''
    return exp(mu + sigma_sq/2)

def variance(mu: array, sigma_sq: array) -> array:
    '''Calculates the variance of n log-normal distributions'''
    return (exp(sigma_sq) - 1) * exp(2*mu+sigma_sq)

## Method of Moments
def find_sigma_sq(mean: array, variance: array) -> array:
    return log(variance/mean**2 + 1)

def find_mu(mean: array, sigma_sq: array) -> array:
    return log(mean) - sigma_sq/2

## Arithmatic
def multiply(mu: array, sigma_sq: array) -> array:
    '''Multiplies n log-normal distributions parameterised by mu and sigma**2 and returns the parameters of a new log-normal distribution'''
    return array([mu.sum(), sigma_sq.sum()])

def sum(mu: array, sigma_sq: array) -> array:
    '''Sums n log-normal distributions parameterised by mu and sigma**2 and returns the parameters of a new log-normal distribution'''
    means = array(mean(mu, sigma_sq).sum())
    variances = array(variance(mu, sigma_sq).sum())

    sigma_sq_hat = find_sigma_sq(means, variances)
    mu_hat = find_mu(means, sigma_sq_hat)

    return array([mu_hat, sigma_sq_hat])
