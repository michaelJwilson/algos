struct Solution;

pub fn max_profit(prices: &Vec<i32>) -> i32 {
    if prices.len() == 0 {
        return 0;
    }

    let mut max_profit = 0;

    for ii in 0..=prices.len() -2 {
        for jj in ii+1..=prices.len() -1 {
	    max_profit = max_profit.max(prices[jj] - prices[ii]);
	}
    }

    max_profit
}

impl Solution {
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        max_profit(&prices)       
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_max_profit_one() {
        let prices = vec![7,1,5,3,6,4];
	let max_profit = max_profit(&prices);

        assert_eq!(max_profit, 5);
    }
    
    #[test]
    pub fn test_max_profit_two() {
        let prices = vec![7,6,4,3,1];
        let max_profit = max_profit(&prices);

        assert_eq!(max_profit, 0);
    }

    #[test]
    pub fn test_max_profit_solution() {
        let prices = vec![7,6,4,3,1];
	let max_profit = Solution::max_profit(prices);

	assert_eq!(max_profit, 0);
    }
}