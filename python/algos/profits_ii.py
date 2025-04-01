import numpy as np
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profits = prices[1:] - prices[:-1]
        profits = np.concatenate([np.zeros(1), profits])
        profits = np.clip(profits, a_min=0., a_max=None).astype(int)
        
        return np.sum(profits)

    
if __name__ == "__main__":
    prices = np.array([7,1,5,3,6,4])
    # prices = np.array([1,2,3,4,5])
    # prices = np.array([7,6,4,3,1])
    
    profits = Solution().maxProfit(prices)
    
    print(prices)
    print(profits)
    
