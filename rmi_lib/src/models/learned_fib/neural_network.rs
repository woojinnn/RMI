use crate::models::*;
use std::convert::{TryFrom, TryInto};

use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn relu(inp: f64) -> f64 {
    match inp {
        x if x < 0.0 => 0.0,
        x if x >= 0.0 => x,
        _ => panic!("relu: type doesn't match"),
    }
}

pub struct NN {
    weights1: Vec<f64>,
    weights2: Vec<f64>,
    biases1: Vec<f64>,
    bias2: f64,
}

impl NN {
    pub fn new() -> NN {
        return NN {
            weights1: Vec::new(),
            weights2: Vec::new(),
            biases1: Vec::new(),
            bias2: 0.0,
        };
    }

    pub fn train<TKey: TrainingKey>(&mut self, dataset: &RMITrainingData<TKey>) {
        let start_idx: usize = 0;
        let end_idx: usize = dataset.len() - 1;

        self.bias2 = dataset.get(start_idx).1 as f64;
        let mut prev_slope: f64 = 0.0;
        // return type of RMITrainingData.get() -> (T: TrainingKey, usize)
        for idx in start_idx..(end_idx - 1) {
            let x1 = dataset.get(idx).0.as_float();
            let y1 = u64::try_from(dataset.get(idx).1).unwrap() as f64;
            let x2 = dataset.get(idx + 1).0.as_float();
            let y2 = u64::try_from(dataset.get(idx + 1).1).unwrap() as f64;

            let cur_slope: f64 = (y2 - y1) / (x2 - x1);
            self.weights1.push((cur_slope - prev_slope).abs());
            self.biases1.push(-(x1 * self.weights1.last().unwrap()));
            self.weights2
                .push(if cur_slope > prev_slope { 1.0 } else { -1.0 });
            prev_slope = cur_slope;
        }
    }

    pub fn inference(&self, input: f64) -> f64 {
        let layer1 = (self.weights1).iter().zip((self.biases1).iter());

        let layer1_result: Vec<f64> = layer1
            .into_iter()
            .map(|x| relu(input.mul_add(*x.0, *x.1)))
            .collect();

        let result: f64 = layer1_result
            .into_iter()
            .zip(self.weights2.iter())
            .map(|x| (x.0) * x.1)
            .sum();
        return result + self.bias2;
    }
    pub fn load(&self, model_path: &String) -> NN {
        let path = Path::new(model_path);
        let display = path.display();

        // open file
        let mut file = match File::open(&path) {
            Err(why) => panic!("couldn't open {}: {}", display, why),
            Ok(file) => file,
        };

        // read contents
        let mut contents: Vec<f64> = Vec::new();
        file.read_f64_into::<LittleEndian>(&mut contents).unwrap();

        // return with NN struct
        let contents_len: usize = contents.len();
        match (contents_len - 1) % 3 {
            0 => {
                let slice_len: usize = (contents_len - 1) / 3;

                let mut w1: Vec<f64> = Vec::new();
                let mut w2: Vec<f64> = Vec::new();
                let mut b1: Vec<f64> = Vec::new();

                w1.extend_from_slice(&contents[0..slice_len]);
                w2.extend_from_slice(&contents[slice_len..2 * slice_len]);
                b1.extend_from_slice(&contents[2 * slice_len..3 * slice_len]);

                return NN {
                    weights1: w1,
                    weights2: w2,
                    biases1: b1,
                    bias2: *contents.last().unwrap(),
                };
            }
            _ => panic!("number of parameter is wierd!"),
        }
    }

    pub fn save(&self, model_path: &String) -> std::io::Result<()> {
        let mut file = File::open(model_path)?;
        for weight1 in &self.weights1 {
            file.write_f64::<LittleEndian>(*weight1)?;
        }
        for weight2 in &self.weights2 {
            file.write_f64::<LittleEndian>(*weight2)?;
        }
        for bias1 in &self.biases1 {
            file.write_f64::<LittleEndian>(*bias1)?;
        }
        file.write_f64::<LittleEndian>(self.bias2)?;
        Ok(())
    }
}
