// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::utils::{common_prefix_size, num_bits};
use crate::models::*;
use log::*;

pub struct RadixModel {
    params: (u8, u8),
}

impl RadixModel {
    pub fn new(data: &ModelData) -> RadixModel {
        if data.len() == 0 {
            return RadixModel { params: (0, 0) };
        }

        let largest_value = data.iter_int_int().map(|(_x, y)| y).max().unwrap();
        let bits = num_bits(largest_value);
        trace!(
            "Radix layer using {} bits, from largest value {} (max layers: {})",
            bits,
            largest_value,
            (1 << (bits + 1)) - 1
        );


        let common_prefix =  common_prefix_size(data);
        trace!("Radix layer common prefix: {}", common_prefix);

        return RadixModel {
            params: (common_prefix, bits),
        };
    }
}

impl Model for RadixModel {
    fn predict_to_int(&self, inp: ModelInput) -> u64 {
        let (left_shift, num_bits) = self.params;

        let as_int: u64 = inp.as_int();
        let res = (as_int << left_shift) >> (64 - num_bits);

        return res;
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }

    fn params(&self) -> Vec<ModelParam> {
        return vec![self.params.0.into(), self.params.1.into()];
    }

    fn code(&self) -> String {
        return String::from(
            "
inline uint64_t radix(uint64_t prefix_length, uint64_t bits, uint64_t inp) {
    return (inp << prefix_length) >> (64 - bits);
}",
        );
    }

    fn function_name(&self) -> String {
        return String::from("radix");
    }
    fn needs_bounds_check(&self) -> bool {
        return false;
    }
    fn restriction(&self) -> ModelRestriction {
        return ModelRestriction::MustBeTop;
    }
}

pub struct RadixTable {
    prefix_bits: u8,
    table_bits: u8,
    hint_table: Vec<u32>
}

impl RadixTable {
    pub fn new(data: &ModelData, bits: u8) -> RadixTable {
        let prefix = common_prefix_size(data);
        let mut hint_table: Vec<u32> = vec![0 ; 1 << bits];

        let mut last_radix = 0;
        for (x, y) in data.iter_int_int() {
            let current_radix = ((x << prefix) >> prefix) >> (64 - (prefix + bits));
            if current_radix == last_radix { continue; }
            assert!(current_radix < hint_table.len() as u64);

            hint_table[current_radix as usize] = y as u32;

            for i in (last_radix + 1)..current_radix {
                hint_table[i as usize] = y as u32;
            }

            last_radix = current_radix;
        }

        for i in (last_radix as usize + 1)..hint_table.len() {
            hint_table[i as usize] = hint_table.len() as u32; 
        }

        return RadixTable {
            prefix_bits: prefix,
            table_bits: bits,
            hint_table: hint_table
        };
    }
}

impl Model for RadixTable {
    fn predict_to_int(&self, inp: ModelInput) -> u64 {
        let as_int: u64 = inp.as_int();
        let prefix = self.prefix_bits;
        let bits = self.table_bits;
        let res = ((as_int << prefix) >> prefix) >> (64 - (bits + prefix));

        let idx = self.hint_table[res as usize] as u64;
        return idx;
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }

    fn params(&self) -> Vec<ModelParam> {
        return vec![self.prefix_bits.into(), self.hint_table.clone().into()];
    }

    fn code(&self) -> String {
        return String::from(format!(
            "
inline uint64_t radix_table(uint64_t prefix_length, uint32_t* table, uint64_t inp) {{
    return table[((inp << prefix_length) >> prefix_length) >> (64 - {})];
}}", self.prefix_bits + self.table_bits
        ));
    }

    fn function_name(&self) -> String {
        return String::from("radix_table");
    }
    fn needs_bounds_check(&self) -> bool {
        return false;
    }
    fn restriction(&self) -> ModelRestriction {
        return ModelRestriction::None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        RadixModel::new(&ModelData::empty());
    }

}
