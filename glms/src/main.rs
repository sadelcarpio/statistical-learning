mod optimizer;
mod poisson_glm;

use crate::poisson_glm::PoissonRegressor;
use polars::prelude::*;
use crate::optimizer::GradientDescent;

fn main() {
    let df: DataFrame = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some("data/path.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    let predictors: DataFrame = df.select(["feat1", "feat2"]).unwrap();
    let response: DataFrame = df.select(["y"]).unwrap();
    let optimizer = GradientDescent::new(0.001);
    let mut model = PoissonRegressor::default().with_optimizer(Box::new(optimizer));
    model.fit(predictors, response, 100);
}
