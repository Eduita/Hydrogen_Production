import numpy as np

class brownian_motion:
    def __init__(self, drift=0, std_dev=1, correlation=0, n_steps=100, seed=None):
        self.drift = drift
        self.std_dev = std_dev
        self.correlation = correlation
        self.n_steps = n_steps
        self.seed = seed

    def correlated_GBM(self, initial_prices):
        np.random.seed(self.seed)

        # Define the mean and covariance of the bivariate normal distribution
        mean = [self.drift, self.drift]
        covariance = [[self.std_dev ** 2, self.std_dev ** 2 * self.correlation],
                      [self.std_dev ** 2 * self.correlation, self.std_dev ** 2]]

        # Generate random samples from the bivariate normal distribution
        samples = np.random.multivariate_normal(mean, covariance, self.n_steps)

        # Calculate the prices using the geometric Brownian motion formula
        prices1 = initial_prices[0] * np.cumprod(np.exp(samples[:, 0]))
        prices2 = initial_prices[1] * np.cumprod(np.exp(samples[:, 1]))

        return prices1, prices2

    def uncorrelated_GBM(self, initial_price):
        np.random.seed(self.seed)

        # Generate random samples from the normal distribution
        samples = np.random.normal(self.drift, self.std_dev, self.n_steps)

        # Calculate the prices using the geometric Brownian motion formula
        prices = initial_price * np.cumprod(np.exp(samples))

        return prices

