mod dataset;
mod algorithm;

use dataset::Dataset;
use algorithm::MultipleLinearRegression;


fn main() {
    let dataset = Dataset::from_csv("data.csv").unwrap();
    let mut mlr = MultipleLinearRegression::new();
    mlr.fit(&dataset);
    match &mlr.params {
        Some(params) => println!("Model parameters: {:?}", params),
        None => println!("Model parameters have not been computed."),
    }
    let predictions = mlr.predict(&dataset.x);
    println!("Predictions: {:?}", predictions);
    let mse = mlr.evaluate(&dataset.x, &dataset.y).unwrap();
    println!("MSE: {:?}", mse);
}
