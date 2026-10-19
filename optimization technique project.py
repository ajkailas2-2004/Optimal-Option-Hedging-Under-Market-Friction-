import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import yfinance as yf
import matplotlib.pyplot as plt

class Solution:
    def __init__(self, s, x, t, r, sig):
        self.s = s
        self.x = x
        self.t = t
        self.r = r
        self.sig = sig

    def d1(self):
        d1_value = (np.log(self.s / self.x) + (self.r + self.sig ** 2 / 2) * self.t) / (self.sig * np.sqrt(self.t))
        return d1_value

    def d2(self):
        d2_value = self.d1() - self.sig * np.sqrt(self.t)
        return d2_value

    def c(self):
        n1 = norm.cdf(self.d1())
        n2 = norm.cdf(self.d2())
        c = self.s * n1 - self.x * np.exp(-(self.r * self.t)) * n2
        return c

    def delta(self):
        return norm.cdf(self.d1())
    def objective_function(self,x_new,c,x_previous,gamma,eta):
        transaction_cost = 0.5*gamma*(x_new - self.delta())**2
        proportional_transaction_cost = c*abs(x_new-x_previous)*self.s
        non_linear_market_impact = eta*(x_new-x_previous)**2
        return transaction_cost + proportional_transaction_cost + non_linear_market_impact

if __name__ == "__main__":
    spot_history = []
    theoretical_shares = []
    market_impacted_shares = []

    apple = yf.Ticker("AAPL")
    expires = apple.options
    expiration = "2026-10-14"
    opt_chain = apple.option_chain(expiration)
    call_strike = list(opt_chain.calls['strike'])
    target_strike = call_strike[len(call_strike) // 2]
    spot_range = np.linspace(target_strike * 0.8, target_strike * 1.2, 50)


bs_model = Solution(s=spot_range[0], x=520.0, t=0.5, r=0.06, sig=0.25)
x_previous = bs_model.delta()
gamma = 1
c_cost = 0.0001
eta = 0.001

for s_val in spot_range:
    bs_model.s = s_val
    target_delta = bs_model.delta()
    result = minimize(fun = lambda x_arr:bs_model.objective_function(x_arr[0],c_cost,x_previous,gamma,eta),
                      x0 =[x_previous],method = 'SLSQP',bounds=[(0.0,1.0)])
    optimal_shares = result.x[0]
    print(f"| Spot: {s_val:.1f} | Target Delta: {target_delta:.4f} | Optimal Shares (x_t): {optimal_shares:.4f}")


    spot_history.append(s_val)
    theoretical_shares.append(target_delta)
    market_impacted_shares.append(optimal_shares)

    x_previous = optimal_shares

#print('Available expiratons:',expires)
#Visualisation
plt.figure(figsize = (10,10))
plt.plot(spot_history,theoretical_shares, label=" without market friction",linestyle="--",color="blue")
plt.plot(spot_history,market_impacted_shares, label="with market friction",linestyle="-.",color="red",linewidth=3)
plt.xlabel("spot/strike price")
plt.ylabel("optimized hedge postion/shares")
plt.title("Karush-Kuhn-Tucker(KKT) used study on: Impact of Transaction Friction in real Market")
plt.legend()
plt.grid(True)
plt.show()









































