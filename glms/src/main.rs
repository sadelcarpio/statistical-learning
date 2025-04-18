mod optimizer;
mod poisson_glm;

use crate::optimizer::GradientDescent;
use crate::poisson_glm::PoissonRegressor;
use polars::prelude::*;

fn main() {
    let df: DataFrame = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("data/dummy_data.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    let predictors: DataFrame = df.select(["feat1", "feat2"]).unwrap();
    let response: DataFrame = df.select(["y"]).unwrap();
    let optimizer = GradientDescent::new(0.001);
    let mut model = PoissonRegressor::default().with_optimizer(Box::new(optimizer));
    let metrics = model.fit(&predictors, &response, 1000);
    println!("Metrics: {:?}", metrics);
    // Prediction on train data:
    let predictions = model.predict(&predictors);
    println!("Ground Truth: {:?}", response);
    println!("Predictions: {:?}", predictions);
    let metrics = model.evaluate(&predictors, &response);
    println!("Metrics: {:?}", metrics);
    println!("Trained params: {:?}", model.params.unwrap());
}
