pub fn leapfrog<F>(
    f: F,                // Function to compute acceleration (force/mass)
    position: f64,       // Initial position
    velocity: f64,       // Initial velocity
    time_step: f64,      // Time step (Δt)
    num_steps: usize,    // Number of steps to integrate
) -> (Vec<f64>, Vec<f64>) // Returns positions and velocities
where
    F: Fn(f64) -> f64,   // Acceleration function: a = f(position)
{
    let mut positions = vec![position];
    let mut velocities = vec![velocity];

    let mut current_position = position;
    let mut current_velocity = velocity;

    for _ in 0..num_steps {
        let half_velocity = current_velocity + 0.5 * time_step * f(current_position);

        current_position += time_step * half_velocity;
        current_velocity = half_velocity + 0.5 * time_step * f(current_position);

        positions.push(current_position);
        velocities.push(current_velocity);
    }

    (positions, velocities)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leapfrog_harmonic_oscillator() {
        // NB define the acceleration for a harmonic oscillator (with unit mass/spring constant): a = -x
        let acceleration = |x: f64| -x;

        let initial_position = 1.0; // x(0) = 1
        let initial_velocity = 0.0; // v(0) = 0
        let time_step = 0.1;        // Δt
        let num_steps = 100;        // Number of steps

        let (positions, velocities) = leapfrog(
            acceleration,
            initial_position,
            initial_velocity,
            time_step,
            num_steps,
        );

        // Verify the results (simple harmonic motion: x(t) ≈ cos(t), v(t) ≈ -sin(t))
        let tolerance = 0.1; // Allowable error due to numerical approximation
        
        for (i, &pos) in positions.iter().enumerate() {
            let time = i as f64 * time_step;
            let expected_position = (time).cos();
            
            assert!(
                (pos - expected_position).abs() < tolerance,
                "Position at step {} is incorrect: got {}, expected {}",
                i,
                pos,
                expected_position
            );
        }
    }
}