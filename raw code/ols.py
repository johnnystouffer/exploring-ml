import numpy as np
import matplotlib.pyplot as plt

class OLS:
    def __init__(self, x=None, y=None):
        self.x = x if x is not None else []
        self.y = y if y is not None else []
        self.slope = None
        self.intercept = None
        self.line = None
        self.line_use = None
        self.p = None
        self.p_use = None
        self.r = None
        self.r_squared = None
        self.rmse = None
        self.rse = None

    def calc_best_fit(self):
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)

        numerator = np.sum(((self.x - x_mean) * (self.y - y_mean)))
        denominator = np.sum((self.x - x_mean) ** 2)

        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean

    def find_variables(self):
        print(f"Slope: {round(self.slope, 2)}")
        print(f"Intercept: {round(self.intercept, 2)}")
        return self.slope, self.intercept

    def lin_reg(self):
        self.line = f'y = {round(self.intercept, 2)} + {round(self.slope, 2)}x'
        self.line_use = self.intercept + self.slope * self.x
        print(f"Our best line of fit is {self.line}")
        return self.line, self.line_use

    def log_reg(self):
        pnum = np.exp(self.intercept + self.slope * self.x)
        pden = 1 + np.exp(self.intercept + self.slope * self.x)
        self.p = f'p = e^({round(self.intercept, 2)} + {round(self.slope, 2)}x) / (1 + e^({round(self.intercept, 2)} + {round(self.slope, 2)}x))'
        self.p_use = pnum / pden
        print(f"Our best line of fit is {self.p}")
        return self.p, self.p_use

    def corr_coef(self):
        n = len(self.x)
        numerator = (n * np.sum(self.x * self.y)) - (np.sum(self.x) * np.sum(self.y))
        denominator = np.sqrt((n * np.sum(self.x ** 2) - (np.sum(self.x) ** 2)) * (n * np.sum(self.y ** 2) - (np.sum(self.y) ** 2)))
        self.r = numerator / denominator
        self.r_squared = self.r ** 2
        print(f"Our R value is {round(self.r, 4)}, and our R^2 value is {round(self.r_squared, 4)}")
        return self.r, self.r_squared

    def results(self):
        print(f"Best Line of Fit: {self.line}")
        print(f"Correlation Coefficient: {round(self.r, 4)}")
        print(f"Goodness of Fit: {round(self.r_squared, 4)}")

    def predict(self, x):
        y = self.intercept + self.slope * x
        return print(f"Predicted y value given {x}: {y}")

    def predict_list(self, values):
        y = [self.intercept + self.slope * i for i in values]
        print(f"Predicted y values given {values}: {y}")
        return y

    def rmse(self):
        n = len(self.x)
        numerator = np.sum((self.line_use - self.y) ** 2)
        self.rmse = np.sqrt(numerator / n)
        print(f"Root Mean Squared Error: {round(self.rmse, 4)}")
        return self.rmse

    def rse(self):
        n = len(self.x)
        numerator = np.sum((self.line_use - self.y) ** 2)
        self.rse = np.sqrt(numerator / (n - 2))
        print(f"Residual Standard Error: {round(self.rse, 4)}")
        return self.rse

    def resid_plot(self):
        plt.figure(figsize=(10, 6))
        plt.title('Residual Plot')
        plt.scatter(self.x, self.y - self.line_use, color='green')
        plt.axhline(0, color='red')
        plt.xlabel('x')
        plt.ylabel('y - y_hat')
        plt.grid()
        plt.show()

    def plot_line(self, x_label, y_label):
        plt.figure(figsize=(10, 6))
        plt.title('Best Line of Fit')
        plt.scatter(self.x, self.y, color='green')
        line = self.intercept + self.slope * self.x
        plt.plot(self.x, line, color='red', label="Line of Best Fit")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid()
        plt.show()