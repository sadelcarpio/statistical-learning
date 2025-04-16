use crate::dataset::Dataset;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray_linalg::Inverse;

pub struct MultipleLinearRegression {
    pub params: Option<Array1<f32>>,
}

impl MultipleLinearRegression {
    pub fn new() -> Self {
        Self { params: None }
    }

    /// Builds the design matrix for regression model, of shape `[n][1 + p]`
    /// # Arguments
    ///
    /// * `x`: Matrix of predictors of shape `[n][p]`
    ///
    /// returns: ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>>
    fn build_design_matrix(x: &Array2<f32>) -> Array2<f32> {
        let ones_column = Array::from_elem((x.nrows(), 1), 1.0);
        concatenate![Axis(1), ones_column, x.view()]
    }

    /// Fits the regression model by using the closed formula:
    /// `b = (((X.T)X)^-1)(X.T)Y`
    ///
    /// # Arguments
    ///
    /// * `dataset`: Dataset struct, containing predictors and response
    ///
    /// returns: ()
    pub fn fit(&mut self, dataset: &Dataset) {
        let x = Self::build_design_matrix(&dataset.x); // n x (p + 1)
        let xt = x.t(); // (p + 1) x n
        let xtx = xt.dot(&x); // (p + 1) x (p + 1)
        let xtx_inv = xtx.inv().expect("Cannot invert matrix. Verify n > p");
        let xty = xt.dot(&dataset.y); // (p + 1) x 1
        self.params = Some(xtx_inv.dot(&xty)); // (p + 1) x 1
    }

    /// Generates a prediction on a batch of predictors given by:
    /// `y_hat = Xb`
    ///
    /// # Arguments
    ///
    /// * `x`: Matrix of predictors, shape `[m][p]`
    ///
    /// returns: ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>
    pub fn predict(&mut self, x: &Array2<f32>) -> Array1<f32> {
        let x = Self::build_design_matrix(&x);
        let params = self
            .params
            .as_ref()
            .expect("Model parameters have not set.");
        x.dot(params)
    }

    /// Calculates the rmse of the prediction of X against the ground truth label
    ///
    /// # Arguments
    ///
    /// * `x`: predictors
    /// * `y`: ground truth labels
    ///
    /// returns: Option<f32>
    pub fn evaluate(&mut self, x: &Array2<f32>, y: &Array1<f32>) -> Option<f32> {
        let pred = self.predict(x);
        Some((y - pred).mapv(|x| x.powi(2)).mean()?.sqrt())
    }
}
