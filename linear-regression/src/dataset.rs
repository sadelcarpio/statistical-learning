use ndarray::prelude::*;
use polars::prelude::*;

pub struct Dataset {
    pub x: Array2<f32>,
    pub y: Array1<f32>,
}

impl Dataset {
    /// Reads the dataset from a csv file populating the values of x and y, being y the last column
    ///
    /// # Arguments
    ///
    /// * `filename`: Path to the csv file
    ///
    /// returns: Result<Dataset, Box<dyn Error, Global>>
    pub fn from_csv(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(filename.into()))
            .unwrap()
            .finish()
            .unwrap();
        let p: usize = df.width() - 1;
        let arr = df.to_ndarray::<Float32Type>(Default::default()).unwrap();
        let x: Array2<f32> = arr.slice(s![.., ..p]).into_owned();
        println!("X: {:?}", x);
        let y: Array1<f32> = arr.column(p).into_owned();
        println!("Y: {:?}", y);
        Ok(Self { x, y })
    }
}
