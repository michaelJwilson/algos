from typing import List

import numpy as np
from numba import njit


@njit
def get_profit(prices):
    min_val = prices[0]
    result = []

    for ii in range(len(prices)):
        result.append(prices[ii] - min_val)
        min_val = min(min_val, prices[ii])

    return np.array(result)


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        prices = np.array(prices)
        profits = get_profit(prices)

        # print(profits)

        return np.max(profits)


if __name__ == "__main__":
    # prices = np.array([7, 1, 5, 3, 6, 4])
    # prices = np.array([1,2,3,4,5])
    prices = np.array([7, 6, 4, 3, 1])

    profits = Solution().maxProfit(prices)

    print(prices)
    print(profits)
