use crate::optimizer::Optimizer;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use polars::prelude::{DataFrame, Float32Type};

// pub struct PoissonRegressor<'a> {
//     params: Option<Array1<f32>>,
//     optimizer: Option<&'a dyn Optimizer>  // Optimizer already exists and its referenced
// }

pub struct PoissonRegressor {
    params: Option<Array2<f32>>,
    optimizer: Option<Box<dyn Optimizer>>, // Box allocates T on the heap, slower.
                                           // dyn specifies any struct that implements Optimizer trait
}

impl Default for PoissonRegressor {
    fn default() -> Self {
        Self {
            params: None,
            optimizer: None,
        }
    }
}

impl PoissonRegressor {
    pub fn new(optimizer: Box<dyn Optimizer>) -> Self {
        Self {
            params: None,
            optimizer: Option::from(optimizer),
        }
    }

    pub fn with_optimizer(self, optimizer: Box<dyn Optimizer>) -> Self {
        Self {
            params: None,
            optimizer: Some(optimizer),
        }
    }

    pub fn fit(&mut self, x: DataFrame, y: DataFrame, num_iters: i32) {
        let x_arr = x.to_ndarray::<Float32Type>(Default::default()).unwrap(); // (n) x (p)
        let y_arr = y.to_ndarray::<Float32Type>(Default::default()).unwrap();
        let design_matrix = concatenate![Axis(1), x_arr, Array::from_elem((x_arr.nrows(), 1), 1.0)]; // (n) x (p+1)
        let mut params_arr = Array::random(
            (design_matrix.ncols(), 1),
            Uniform::new(-1.0, 1.0),
        );
        let optimizer = self.optimizer.as_ref().unwrap();
        for iter in 0..num_iters {
            let z = design_matrix.dot(&params_arr);
            let y_pred = z.clone().mapv(f32::exp);
            let loss = (&y_pred - &y_arr * z).sum_axis(Axis(0));
            println!("Loss at step {}: {}", iter, loss);
            let error = &y_pred - &y_arr;
            let gradient = design_matrix.t().dot(&error);
            params_arr = optimizer.step(&params_arr, &gradient);
        }
        self.params = Some(params_arr);
    }
}
