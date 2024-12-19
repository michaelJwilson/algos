// TODO: remove this when you're done with your implementation.
// #![allow(unused_variables, dead_code)]

fn transpose(matrix: [[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let mut result: [[i32; 3]; 3] = [[0; 3]; 3];

    for i in 0..=2{
        for j in 0..=2{
            result[i][j] = matrix[j][i];
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let matrix = [
    	    [101, 102, 103], //
            [201, 202, 203],
            [301, 302, 303],
    	];

	let transposed = transpose(matrix);
    
        println!("matrix: {:#?}", matrix);
	    println!("transposed: {:#?}", transposed);
    
        assert_eq!(
    	    transposed,
            [
		[101, 201, 301], //
            	[102, 202, 302],
            	[103, 203, 303],
            ]
        );
    }
}