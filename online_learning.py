import os
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class BetaBinominal(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def update(self, m, n):
        """ Returns a new model after observing data """
        return BetaBinominal(self.a + m, self.b + n)

    def pdf(self, x):
        """ Returns the PDF of the prior function at x """
        return stats.beta.pdf(x, self.a, self.b)

    def cdf(self, x):
        """ Returns the CDF of the prior function at x """
        return stats.beta.cdf(x, self.a, self.b)

    def posterior(self, l, r):
        """ Returns the credible interval on (l, r) """
        if l > r:
            return 0.0

        return self.cdf(u)-self.cdf(l)

    def prior_mean(self):
        """ Returns the prior mean """
        return self.a / (self.a + self.b)

    def prior_variance(self):
        """ Returns the prior variance """
        return (self.a * self.b) / ((self.a+self.b)**2 * (self.a + self.b + 1))

    def plot_prior(self, l=0.0, u=1.0):
        x = np.linspace(u, l, 1001)
        y = stats.beta.pdf(x, self.a, self.b)
        y = y / y.sum()
        plt.plot(x, y)
        plt.xlim((l,u))
        plt.show()

    def print_status(self):
        print("Parameters of beta distribution:")
        print("a: {}  b: {}".format(self.a, self.b))
        print("mean: %f" % self.prior_mean())
        print("variance: %f" % self.prior_variance())
        print("---------------------------------\n")

def read_binary(file):
    while True:
        data = file.readline()
        if not data:
            break

        yield data

def online_learning(file_path, model, plot):

    if model.__class__.__name__ == 'BetaBinominal':
        with open(file_path) as f:
            for line in read_binary(f):
                print("Data: {}".format(line[:-1]))
                N = len(line) - 1
                m = np.sum([1 if line[d]=='1' else 0 for d in range(len(line)-1)])

                print("[+] Binomial likelihood: {}/{}\n".format(m, N))
                model = model.update(m, N - m)
                model.print_status()

                if plot:
                    model.plot_prior()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='specify the data file location')
    parser.add_argument('-a', type=int, help='specify the beta parameter - a', default=2)
    parser.add_argument('-b', type=int, help='specify the beta parameter - b', default=2)
    parser.add_argument('-p', '--plot', action='store_true', help='Whether plot the prior distribution or not', default=False)
    args = parser.parse_args()

    file_path = os.path.join('data', 'binary.txt') if args.file == None else args.file

    model = BetaBinominal(args.a, args.b)
    model.print_status()

    if args.plot:
        model.plot_prior()

    online_learning(file_path, model, args.plot)
