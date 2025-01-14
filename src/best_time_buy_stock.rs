struct Solution;

// NB naive pairwise test;  works for relatively small price list.
pub fn naive_max_profit(prices: &Vec<i32>) -> i32 {
    if prices.len() < 2 {
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

pub fn backward_max_profit(prices: &Vec<i32>) -> i32 {
    if prices.len() < 2 {
        return 0;
    }

    let last_idx = prices.len() -1;
    let mut max_profit: i32 = 0;
    let	mut max_prices: Vec<i32> = vec![0; prices.len() as usize];

    max_prices[last_idx] = prices[last_idx];

    // NB backward
    for jj in (0..last_idx).rev() {
        max_prices[jj] = max_prices[jj+1].max(prices[jj]);
    }

    println!("{:?}", max_prices);
    
    // NB forward.  Cannot buy on the last day.
    for ii in 0..=prices.len() -2 {
        max_profit = max_profit.max(max_prices[ii+1] - prices[ii]);
    }
    
    max_profit
}

fn max_profit(prices: Vec<i32>) -> i32 {
    if prices.is_empty() {
        return 0;
    }

    let mut min_price = prices[0];
    let mut max_profit = 0;

    for &price in &prices[1..] {
        if price < min_price {
            min_price = price;
        } else {
            let profit = price - min_price;
            if profit > max_profit {
                max_profit = profit;
            }
        }
    }

    max_profit
}

impl Solution {
    pub fn max_profit(prices: Vec<i32>) -> i32 {
        max_profit(prices)
    }
}


#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings" cargo test test_max_profit_one -- --nocapture
    use super::*;

    #[test]
    pub fn test_max_profit_one() {
        let prices = vec![7,1,5,3,6,4];

	println!("{:?}", prices);

	let max_profit = max_profit(prices);

        assert_eq!(max_profit, 5);
    }
    
    #[test]
    pub fn test_max_profit_two() {
        let prices = vec![7,6,4,3,1];

	println!("{:?}", prices);

	let max_profit = max_profit(prices);

        assert_eq!(max_profit, 0);
    }

    #[test]
    pub fn test_max_profit_three() {
    	let prices = vec![1,4,2];

	println!("{:?}", prices);

	let max_profit = max_profit(prices);

	assert_eq!(max_profit, 3);
    }

    #[test]
    pub fn test_max_profit_four() {
    	let prices = vec![3,2,6,5,0,3];

	println!("{:?}", prices);

	let max_profit = max_profit(prices);
	
	assert_eq!(max_profit, 4);
    }

    #[test]
    pub fn test_max_profit_five() {
        let prices = vec![2,1];

	println!("{:?}", prices);

	let max_profit = max_profit(prices);

	assert_eq!(max_profit, 0);
    }

    #[test]
    pub fn test_max_profit_solution() {
        let prices = vec![7,6,4,3,1];
	let max_profit = Solution::max_profit(prices);

	assert_eq!(max_profit, 0);
    }
}