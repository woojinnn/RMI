// < begin copyright >
// Copyright Ryan Marcus 2020
//
// See root directory of this project for license terms.
//
// < end copyright >

use crate::models::*;
use std::fs;

use std::convert::{TryFrom, TryInto};

mod neural_network;

fn clip(inp: u64, prefix: u64) -> usize {
    let mask: u64 = (1 << prefix) - 1;
    let val: u64 = (inp & mask) >> prefix;
    return u64::try_into(val).unwrap();
}

pub struct LearnedFIB {
    prefix: u64,
    neural_networks: Vec<neural_network::NN>,
    max_error: u64,
}

impl LearnedFIB {
    pub fn new<T: TrainingKey>(
        data: &RMITrainingData<T>,
        threshold: u64,
        prefix: u64,
    ) -> LearnedFIB {
        let mut neural_networks: Vec<neural_network::NN> = Vec::new();
        for _ in 1..(1 << prefix) {
            neural_networks.push(neural_network::NN::new());
        }

        // train
        let mut prev_prefix: usize = 0;
        let mut from: usize = 0;
        let mut to: usize = 0;
        for datum in data.iter() {
            let (key, _) = datum;
            let cur_prefix: usize = clip(key.as_uint(), prefix);

            if prev_prefix != cur_prefix {
                LearnedFIB::train_subset(
                    data,
                    from,
                    to,
                    &mut neural_networks[cur_prefix],
                    threshold as f64,
                );
                prev_prefix = cur_prefix;
                from = to + 1;
            }
            to = to + 1;
        }

        // check_error
        let mut max_error = 0;
        for datum in data.iter() {
            let (key, value) = datum;
            let answer = u64::try_from(value).unwrap() as f64;
            let nn_idx: usize = clip(key.as_uint(), prefix);
            let predicted: f64 = neural_networks[nn_idx].inference(key.as_float());
            let err: u64 = if predicted > answer {
                (predicted - answer) as u64
            } else {
                (answer - predicted) as u64
            };
            if err > max_error {
                max_error = err;
            }
        }

        //return
        return LearnedFIB {
            prefix: prefix,
            neural_networks: neural_networks,
            max_error: max_error,
        };
    }

    // same as derive_boundaries() and train nerual network
    fn train_subset<T: TrainingKey>(
        data: &RMITrainingData<T>,
        from: usize,
        to: usize,
        nn: &mut neural_network::NN,
        threshold: f64,
    ) {
        let mut boundary: Vec<(T, usize)> = Vec::new();
        let mut l: usize = from;
        for r in (from + 2)..to {
            // handling duplicate keys
            if data.get(l).0 == data.get(r).0 {
                continue;
            }
            if data.get(l).0 == data.get(r - 1).0 {
                continue;
            }

            let (key_l, val_l) = data.get(l);
            let (key_r, val_r) = data.get(r);
            let x_l = key_l.as_float();
            let y_l = u64::try_from(val_l).unwrap() as f64;
            let x_r = key_r.as_float();
            let y_r = u64::try_from(val_r).unwrap() as f64;

            // Derive a line's (slope, bias) passing through (x_l, l) and (x_r, r)
            let a: f64 = (y_r - y_l) / (x_r - x_l);
            let b: f64 = y_l - a * x_l;

            // Examine the error between x_(l+1) and x_(r-1)
            for i in (l + 1)..(r - 1) {
                let (key_i, val_i) = data.get(i);
                let x_i = key_i.as_float();
                let y_i = u64::try_from(val_i).unwrap() as f64;

                // compute the y-value on the line for the x-value of x_i
                let p: f64 = a * x_i + b;

                let err: f64 = if p > y_i { p - y_i } else { y_i - p };
                if err > threshold {
                    boundary.push(data.get(r - 1));
                    l = r - 1;
                    break;
                }
            }
        }

        // insert last point if not inserted
        let last_data = data.get_key(data.len() - 1);
        let last_boundary = boundary.last().unwrap().0;
        if last_data != last_boundary {
            boundary.push(data.get(data.len() - 1));
        }

        nn.train(&RMITrainingData::new(Box::new(boundary)));
    }

    // save NN
    fn save(&self, path: String) -> std::io::Result<()> {
        for i in 0..(1 << self.prefix) {
            let file_name: String = String::from("nn_") + &i.to_string();
            fs::File::create(&file_name)?;
            self.neural_networks[i].save(&file_name)?;
        }
        Ok(())
    }
}

impl Model for LearnedFIB {
    fn predict_to_float(&self, inp: &ModelInput) -> f64 {
        let nn_idx: usize = clip(inp.as_int(), self.prefix);
        return self.neural_networks[nn_idx].inference(inp.as_float());
    }

    fn predict_to_int(&self, inp: &ModelInput) -> u64 {
        return f64::max(0.0, self.predict_to_float(inp).floor()) as u64;
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }

    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }

    // TODO
    fn params(&self) -> Vec<ModelParam> {
        return Vec::new();
    }

    // TODO
    fn code(&self) -> String {
        return String::from(
            "
inline uint64 learned_fib(char *mod_path, double inp) {
    return std::fma(beta, inp, alpha);
}",
        );
    }

    fn function_name(&self) -> String {
        return String::from("LearnedFIB");
    }

    fn needs_bounds_check(&self) -> bool {
        return true;
    }

    fn restriction(&self) -> ModelRestriction {
        return ModelRestriction::MustBeBottom;
    }

    fn error_bound(&self) -> Option<u64> {
        return Some(self.max_error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_linear_spline1() {
    //     let md = ModelData::IntKeyToIntPos(vec![(1, 2), (2, 3), (3, 8)]);

    //     let lin_mod = LinearSplineModel::new(&md);

    //     assert_eq!(lin_mod.predict_to_int(1.into()), 2);
    //     assert_eq!(lin_mod.predict_to_int(3.into()), 8);
    // }
}
