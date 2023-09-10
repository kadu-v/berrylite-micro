pub mod relu;

use crate::kernel::utils::quantization::quantize;
use crate::micro_array::ArrayElem;
use crate::micro_erros::{BLiteError, Result};
use relu::relu;

pub fn get_activation<T: ArrayElem<T>>(op_code: i32) -> Option<fn(T) -> T> {
    match op_code {
        1 => Some(relu),
        _ => None,
    }
}

pub fn calculate_fused_activation_range_quantized(
    scale: f32,
    zero_point: i32,
    op: i32,
) -> Result<(i32 /* activtion_min */, i32 /* activation_max */)> {
    let mut activation_min = core::i8::MIN as i32;
    let mut activation_max = core::i8::MAX as i32;

    // Note that the I only consider the type of tensor is int8
    match op {
        1 /* Relu */  => {
            let q_min = quantize(scale, zero_point, 0.0)?;
            activation_min = core::cmp::max(activation_min, q_min);
        },
        2 /* ReluN1To1 */ => {
            let q_min = quantize(scale, zero_point, -1.0)?;
            let q_max = quantize(scale, zero_point, 1.0)?;
            activation_min = core::cmp::max(activation_min, q_min);
            activation_max = core::cmp::min(activation_max, q_max);
        }
        3 /* Relu6 */ => {
            let q_min = quantize(scale, zero_point, 0.0)?;
            let q_max = quantize(scale, zero_point, 6.0)?;
            activation_min = core::cmp::max(activation_min, q_min);
            activation_max = core::cmp::min(activation_max, q_max);
        },
        0 /* None */ |4 /* Tanh */ | 5 /*SignBit */ | 6 /* Sigmoid */ => {/* do nothing */},
        _ => {
            return Err(BLiteError::NotFoundFusedActivation(op))
        }
    }

    Ok((activation_min, activation_max))
}
