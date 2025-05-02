#[inline(always)]
pub fn acceleration(x: f64) -> f64 {
    -x
}

pub struct State {
    position: f64,
    velocity: f64,
}

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
    let mut positions = vec![0.0; 1 + num_steps];
    let mut velocities = vec![0.0; 1 + num_steps];

    positions[0] = position;
    velocities[0] = velocity;

    for i in 0..num_steps {
        let half_velocity = velocities[i] + 0.5 * time_step * f(positions[i]);
        
        positions[i + 1] = positions[i] + time_step * half_velocity;
        velocities[i + 1] = half_velocity + 0.5 * time_step * f(positions[i + 1]);
    }

    (positions, velocities)
}

pub fn get_leapfrog_fixture() -> (f64, f64, f64, usize){
    // NB define the acceleration for a harmonic oscillator (with unit mass/spring constant): a = -x
    let initial_position = 1.0; // x(0) = 1
    let initial_velocity = 0.0; // v(0) = 0

    let time_step = 0.1;        // Δt
    let num_steps = 1_00;      // Number of steps

    (initial_position, initial_velocity, time_step, num_steps)
}

pub fn leapfrog_optimized<F>(
    f: F,
    initial_position: f64,
    initial_velocity: f64,
    time_step: f64,
    num_steps: usize,
) -> Vec<State>
where
    F: Fn(f64) -> f64,
{
    let mut states = Vec::with_capacity(num_steps + 1);
    states.push(State {
        position: initial_position,
        velocity: initial_velocity,
    });

    for _ in 0..num_steps {
        let last = states.last().unwrap();
        let half_velocity = last.velocity + 0.5 * time_step * f(last.position);
        let new_position = last.position + time_step * half_velocity;
        let new_velocity = half_velocity + 0.5 * time_step * f(new_position);

        states.push(State {
            position: new_position,
            velocity: new_velocity,
        });
    }

    states
}


#[cfg(test)]
mod tests {
    // cargo test leapfrog -- test_leapfrog_harmonic_oscillator --nocapture
    use super::*;

    #[test]
    fn test_leapfrog_harmonic_oscillator() {
        let (initial_position, initial_velocity, time_step, num_steps) = get_leapfrog_fixture();

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
            let exp = (time).cos();

            //  println!("{:.3}\t{:+.3}\t{:+.3}", time, exp, pos);

            assert!(
                (pos - exp).abs() < tolerance,
                "Position at step {} is incorrect: got {}, expected {}",
                i,
                pos,
                exp
            );
        }
    }
}