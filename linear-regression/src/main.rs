use ndarray::prelude::*;
use polars::prelude::*;

struct Dataset {
    x: Arc<Array2<f32>>,
    y: Arc<Array1<f32>>,
}

impl Dataset {
    fn new(x: Arc<Array2<f32>>, y: Arc<Array1<f32>>) -> Dataset {
        Dataset { x, y }
    }

    fn from_csv(filename: &str) -> Dataset {
        let df = CsvReadOptions::default().try_into_reader_with_file_path(Some(filename.into())).unwrap().finish().unwrap();
        let p: usize = df.width() - 1;
        let arr = df.to_ndarray::<Float32Type>(Default::default()).unwrap();
        let x: Array2<f32> = arr.slice(s![.., ..p]).into_owned();
        let y: Array1<f32> = arr.column(p).into_owned();
        return Dataset { x: Arc::new(x), y: Arc::new(y) };
    }
}

fn main() {
    let dataset = Dataset::from_csv("data.csv");
    println!("{:?}", dataset.y);
    println!("{:?}", dataset.x);
}
