// NB a 3x3 matrix  == 3 rows of 3-element (i32) vectors.
fn transpose(matrix: [[i32; 3]; 3]) -> [[i32; 3]; 3] {
    // NB Initialize with zeros.
    let mut result: [[i32; 3]; 3] = [[0; 3]; 3];

    // NB Inclusive, i in {0, 1, 2}.
    for (i, row) in matrix.iter().enumerate() {
        for (j, &item) in row.iter().enumerate() {
            result[j][i] = item;
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
