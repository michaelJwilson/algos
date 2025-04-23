use ndarray::Array2;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::mem;

const RAND_SEED: u64 = 42;

struct Solution;

#[derive(Debug)]
struct GameOfLife {
    pub num_rows: usize,
    pub num_cols: usize,
    pub lattice: Array2<i32>,
    pub scratch: Array2<i32>,
}

impl GameOfLife {
    const NEIGHBOR_OFFSETS: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(RAND_SEED);
        let lattice: Array2<i32> =
            Array2::from_shape_fn((num_rows, num_cols), |_| rng.random_range(0..2));

        let scratch: Array2<i32> = Array2::zeros(lattice.raw_dim());
        // let scratch: Array2<i32> = lattice.clone();

        GameOfLife {
            num_rows,
            num_cols,
            lattice,
            scratch,
        }
    }

    pub fn from_array(lattice: Array2<i32>) -> Self {
        let (num_rows, num_cols) = lattice.dim();

        let scratch: Array2<i32> = Array2::zeros(lattice.raw_dim());
        //  let scratch: Array2<i32> = lattice.clone();

        GameOfLife {
            num_rows,
            num_cols,
            lattice,
            scratch,
        }
    }

    pub fn new_cell_state(live: i32, num_neighbors: i32) -> i32 {
        if live > 0 {
            if num_neighbors < 2 {
                0
            } else if num_neighbors < 4 {
                1
            } else {
                0
            }
        } else {
            match num_neighbors {
                3 => 1,
                _ => 0,
            }
        }
    }

    fn valid_indices(&self, i: i32, j: i32) -> bool {
        (i >= 0) && (i < self.num_rows as i32) && (j >= 0) && (j < self.num_cols as i32)
    }

    pub fn update(&mut self) {
        for ((i, j), &value) in self.lattice.indexed_iter() {
            // NB calculate the number of neighbors.
            let mut num_neighbors: i32 = 0;

            /*
            for ishift in 0..=2 {
                for jshift in 0..=2 {
                    if (ishift == 1) && (jshift == 1) {
                        continue;
                    }

                    let new_row_index = (i + ishift) as i32 - 1;
                    let new_col_index = (j + jshift) as i32 - 1;

                    if self.valid_indices(new_row_index, new_col_index) {
                        num_neighbors +=
                            self.lattice[(new_row_index as usize, new_col_index as usize)];
                    }
                }
            }
            */

            for &(di, dj) in &Self::NEIGHBOR_OFFSETS {
                let new_row_index = i as i32 + di;
                let new_col_index = j as i32 + dj;

                if self.valid_indices(new_row_index, new_col_index) {
                    num_neighbors += self.lattice[(new_row_index as usize, new_col_index as usize)];
                }
            }

            // NB update scrate with the new cell state - lattice itself must be preserved.
            self.scratch[(i, j)] = GameOfLife::new_cell_state(value, num_neighbors);
        }

        self.lattice = mem::take(&mut self.scratch);
        // self.lattice = self.scratch.clone();
    }
}

impl Solution {
    pub fn game_of_life(board: &mut [Vec<i32>]) {
        let num_rows = board.len();
        let num_cols = board[0].len();

        let array = Array2::from_shape_fn((num_rows, num_cols), |(i, j)| board[i][j]);

        let mut game = GameOfLife::from_array(array);

        game.update();

        for (i, row) in board.iter_mut().enumerate().take(num_rows) {
            for (j, cell) in row.iter_mut().enumerate().take(num_cols) {
                *cell = game.lattice[(i, j)];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // cargo test game_of_life -- --nocapture
    //
    // See:
    //     https://leetcode.com/problems/game-of-life/description/?envType=study-plan-v2&envId=top-interview-150
    //
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_game_of_life_new() {
        let new = GameOfLife::new(5, 4);

        assert_eq!(new.lattice.shape(), &[5, 4]);

        for &value in new.lattice.iter() {
            assert!(value == 0 || value == 1);
        }
    }

    #[test]
    fn test_game_of_life_from_array() {
        let array =
            Array2::from_shape_vec((4, 3), vec![0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]).unwrap();
        let exp = array.clone();

        let new = GameOfLife::from_array(array);

        for ((i, j), &value) in new.lattice.indexed_iter() {
            assert_eq!(value, exp[(i, j)]);
        }
    }

    #[test]
    fn test_game_of_life_new_cell_state() {
        let mut exp: HashMap<(i32, i32), i32> = HashMap::new();

        exp.insert((1, 0), 0);
        exp.insert((1, 1), 0);
        exp.insert((1, 2), 1);
        exp.insert((1, 3), 1);
        exp.insert((1, 4), 0);

        exp.insert((0, 0), 0);
        exp.insert((0, 1), 0);
        exp.insert((0, 2), 0);
        exp.insert((0, 3), 1);
        exp.insert((0, 4), 0);

        for live in 0..=1 {
            for num_neighbors in 0..=4 {
                assert_eq!(
                    GameOfLife::new_cell_state(live, num_neighbors),
                    *exp.get(&(live, num_neighbors)).unwrap()
                )
            }
        }
    }

    #[test]
    fn test_game_of_life_update() {
        let array =
            Array2::from_shape_vec((4, 3), vec![0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]).unwrap();
        let exp = Array2::from_shape_vec((4, 3), vec![0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]).unwrap();

        let mut new = GameOfLife::from_array(array);
        new.update();

        // println!("{:?}", new.lattice);

        for ((i, j), &value) in new.lattice.indexed_iter() {
            assert_eq!(value, exp[(i, j)]);
        }
    }

    #[test]
    fn test_game_of_life_solution() {
        let mut board = vec![vec![0, 1, 0], vec![0, 0, 1], vec![1, 1, 1], vec![0, 0, 0]];

        Solution::game_of_life(&mut board);

        for row in board {
            println!("{:?}", row);
        }
    }

    #[test]
    fn test_game_of_life_solution_two() {
        let mut board = vec![vec![1, 1], vec![1, 0]];

        let exp = vec![vec![1, 1], vec![1, 1]];

        Solution::game_of_life(&mut board);

        assert_eq!(board[0][0], exp[0][0]);
        assert_eq!(board[0][1], exp[0][1]);
        assert_eq!(board[1][0], exp[1][0]);
        assert_eq!(board[1][1], exp[1][1]);
    }
}
