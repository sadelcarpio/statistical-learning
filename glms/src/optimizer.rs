use ndarray::Array2;

pub trait Optimizer {
    fn step(&self, params: &Array2<f32>, gradient: &Array2<f32>) -> Array2<f32>;
}

pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    pub(crate) fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate
        }
    }
}

impl Optimizer for GradientDescent {
    fn step(&self, params: &Array2<f32>, gradient: &Array2<f32>) -> Array2<f32> {
        params - self.learning_rate * gradient
    }
}
