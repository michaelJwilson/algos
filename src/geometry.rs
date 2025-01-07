// NB Calculate the magnitude of a 3D vector by summing the squares of its coordinates
//    & taking the square root.  Use the `sqrt()` method to calculate the square
//    root, like `v.sqrt()`.
fn magnitude(vector: &[f64; 3]) -> f64 {
   let mut mag_sq = 0.0;

   for vv in vector {
       mag_sq += vv * vv;
   }  
   
   mag_sq.sqrt()
}

// NB Normalize a vector by calculating its magnitude and dividing all of its
// coordinates by that magnitude. In place.
fn normalize(vector: &mut [f64; 3]) {
   let mag = magnitude(vector);
   
   for vv in vector {
      *vv /= mag;
   }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magnitude() {
        println!("Magnitude of a unit vector: {}", magnitude(&[0.0, 1.0, 0.0]));

	let mut v = [1.0, 2.0, 9.0];

	// NB pass by reference to retain ownership of v to main.
        println!("Magnitude of {v:?}: {}", magnitude(&v));
    
	normalize(&mut v);
    
	println!("Magnitude of {v:?} after normalization: {}", magnitude(&v));
    }
}