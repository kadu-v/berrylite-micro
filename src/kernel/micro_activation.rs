use crate::kernel::utils::quantization::quantize;
use crate::micro_array::ArrayElem;
use crate::micro_errors::{BLiteError, Result};
use num_traits::{AsPrimitive, FromPrimitive};

#[inline(always)]
pub(crate) fn activation_with_min_max<T: ArrayElem<T>>(
    x: T,
    activation_min: T,
    activation_max: T,
) -> T {
    let mut ret = x;
    if ret < activation_min {
        ret = activation_min;
    }

    if ret > activation_max {
        ret = activation_max;
    }

    return ret;
}

pub(crate) fn calculate_fused_activation_range<T: ArrayElem<T>>(
    op: i32,
) -> Result<(T /* activtion_min */, T /* activation_max */)> {
    match op {
        0 => Ok((T::MIN, T::MAX)),
        1 => Ok((FromPrimitive::from_f32(0.).unwrap(), T::MAX)),
        _ => Err(BLiteError::NotFoundFusedActivation(op)),
    }
}

pub(crate) fn calculate_fused_activation_range_quantized<T: ArrayElem<T>>(
    scale: f32,
    zero_point: i32,
    op: i32,
) -> Result<(i32 /* activtion_min */, i32 /* activation_max */)> {
    let mut activation_min = AsPrimitive::<i32>::as_(T::MIN);
    let mut activation_max = AsPrimitive::<i32>::as_(T::MAX);

    // Note that I only consider the type of tensor is int8
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
        0 /* None */ | 4 /* Tanh */ | 5 /*SignBit */ | 6 /* Sigmoid */ => {/* do nothing */},
        _ => {
            return Err(BLiteError::NotFoundFusedActivation(op))
        }
    }

    Ok((activation_min, activation_max))
}
