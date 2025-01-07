use ndarray::{array, Array1};
use argmin::core::{observers::ObserverMode, Executor, State, Error, CostFunction, Gradient, Hessian};
use argmin::solver::neldermead::NelderMead;
use argmin_observer_slog::SlogLogger;
use argmin_testfunctions::{
    rosenbrock, rosenbrock_derivative, rosenbrock_hessian
};

struct Rosenbrock {}

impl CostFunction for Rosenbrock {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
         Ok(rosenbrock(&p.to_vec()))
    }
}

impl Gradient for Rosenbrock {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
        Ok(rosenbrock_derivative(p))
    }
}

impl Hessian for Rosenbrock {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, p: &Self::Param) -> Result<Self::Hessian, Error> {
        Ok(rosenbrock_hessian(p))
    }
}


#[test]
fn test_rosenbrock() -> Result<(), Error> {
    let cost = Rosenbrock {};

    let solver = NelderMead::new(vec![
        array![-1.0, 3.0],
        array![2.0, 1.5],
        array![2.0, -1.0],
    ])
    .with_sd_tolerance(0.0001)?;

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(100))
	// .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()?;
	
    println!("{res}");

    let best = res.state().get_best_param().unwrap();
    let best_cost = res.state().get_best_cost();

    let termination_status = res.state().get_termination_status();
    let termination_reason = res.state().get_termination_reason();

    let duration = res.state().get_time().unwrap();
    let num_iters = res.state().get_iter();

    let function_evals = res.state().get_func_counts();

    Ok(())
}