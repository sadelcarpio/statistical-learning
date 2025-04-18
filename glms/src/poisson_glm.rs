use crate::optimizer::Optimizer;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use polars::prelude::{DataFrame, Float32Type};
use std::collections::HashMap;

// pub struct PoissonRegressor<'a> {
//     params: Option<Array1<f32>>,
//     optimizer: Option<&'a dyn Optimizer>  // Optimizer already exists and its referenced
// }

pub struct PoissonRegressor {
    pub params: Option<Array2<f32>>,
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

    pub fn fit(&mut self, x: &DataFrame, y: &DataFrame, num_iters: i32) -> HashMap<&str, f32> {
        let x_arr = x.to_ndarray::<Float32Type>(Default::default()).unwrap(); // (n) x (p)
        let y_arr = y.to_ndarray::<Float32Type>(Default::default()).unwrap();
        let design_matrix = concatenate![Axis(1), x_arr, Array::from_elem((x_arr.nrows(), 1), 1.0)]; // (n) x (p+1)
        let mut params_arr = Array::random((design_matrix.ncols(), 1), Uniform::new(-1.0, 1.0));
        let optimizer = self.optimizer.as_ref().unwrap();
        for iter in 0..num_iters {
            let z = design_matrix.dot(&params_arr);
            let y_pred = z.mapv(f32::exp);
            let error = &y_pred - &y_arr;
            let gradient = design_matrix.t().dot(&error);
            params_arr = optimizer.step(&params_arr, &gradient);
            let neg_log_likelihood = (&y_pred - &y_arr * z).sum();
            let loss = (&y_pred - &y_arr).mapv(|base| base.powi(2)).sum() / (y_pred.nrows() as f32);
            println!("RMSE at step {}: {}", iter + 1, loss);
            println!("Log loss at step {}: {}", iter + 1, neg_log_likelihood);
        }
        self.params = Some(params_arr);
        let final_metrics = self.evaluate(x, y);
        final_metrics
    }

    pub fn predict(&self, x: &DataFrame) -> Array2<f32> {
        let x_arr = x.to_ndarray::<Float32Type>(Default::default()).unwrap();
        let design_matrix = concatenate![Axis(1), x_arr, Array::from_elem((x_arr.nrows(), 1), 1.0)];
        let params_arr = self
            .params
            .as_ref()
            .expect("Params not initialized! Call fit() method first");
        design_matrix.dot(params_arr).mapv(f32::exp)
    }

    pub fn evaluate(&self, x: &DataFrame, y: &DataFrame) -> HashMap<&str, f32> {
        let y_arr = y.to_ndarray::<Float32Type>(Default::default()).unwrap();
        let y_pred = self.predict(x);
        let loss = (&y_pred - &y_arr).mapv(|base| base.powi(2)).sum() / (y_pred.nrows() as f32);
        let nll = (&y_pred * (1.0f32 - &y_pred.mapv(f32::ln))).sum();
        HashMap::from([("rmse", loss), ("nll_loss", nll)])
    }
}
