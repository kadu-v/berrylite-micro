use crate::micro_erros::BLiteError::{self, NotMatchScale};
use crate::micro_erros::Result;

// FullyConnectedParamsQuantized: https://github.com/kadu-v/tflite-micro-sample/blob/0f674d38fc8becd90fbd943fb7e7c49f808a7019/tensorflow/lite/micro/kernels/fully_connected_common.cc#L34
// OpDataFullyConnected: https://github.com/kadu-v/tflite-micro-sample/blob/0f674d38fc8becd90fbd943fb7e7c49f808a7019/tensorflow/lite/micro/kernels/fully_connected.h#L26
// CalculateOpDataFullyConnected: https://github.com/kadu-v/tflite-micro-sample/blob/0f674d38fc8becd90fbd943fb7e7c49f808a7019/tensorflow/lite/micro/kernels/fully_connected_common.cc#L62-L63

/// This function calculates a multiplier following the equation:
/// M = S_1 * S_2 / S_3 (= input_scale * filter_scale / output_scale)
/// see this paper: https://arxiv.org/abs/1712.05877
pub fn get_quantized_convolution_multiplier(
    input_scale: f32,
    filter_scale: f32,
    output_scale: f32,
    bias_scale: Option<f32>,
) -> Result<f64> {
    let input_product_scale = (input_scale * filter_scale) as f64;
    let output_scale = output_scale as f64;
    if let Some(bias_scale) = bias_scale {
        let scale_diff = (input_product_scale - bias_scale as f64).abs();
        if !(scale_diff / output_scale <= 0.02) {
            return Err(NotMatchScale(scale_diff / output_scale));
        }
    }

    let multiplier = input_product_scale / output_scale;
    return Ok(multiplier);
}

/// This function calculates an integer multiplier and an integer shift offset from a real multiplier
/// Note that real_multiplier is included in the interval (0, 1)
/// Note also that an integer multiplier is represented as a fixed point float number
/// Each variable meets the following equations.
/// (1) r = q * shift
/// (2) q_fixed = [fixed point number]
pub fn quantize_multiplier(real_multiplier: f64) -> Result<(i32, i32)> {
    let (q, mut shift) = libm::frexp(real_multiplier);

    let mut q_fixed = (q * (1u64 << 31) as f64).round() as i64;
    if !(q_fixed <= (1 << 31)) {
        return Err(BLiteError::InCompatibleCasting);
    }

    if q_fixed == (1 << 31) {
        q_fixed /= 2;
        shift += 1;
    }

    if !(q_fixed <= core::i32::MAX as i64) {
        return Err(BLiteError::InCompatibleCasting);
    }

    if shift < -31 {
        shift = 0;
        q_fixed = 0;
    }

    // single rounding multiply_by_quantized_multiplier
    if shift > 30 {
        shift = 30;
        q_fixed = (1 << 31) - 1;
    }
    let quantized_multiplier = q_fixed as i32;
    Ok((quantized_multiplier, shift))
}

pub fn multiply_by_quantized_multiplier(
    x: i32,
    quantized_multiplier: i32,
    shift: i32,
) -> Result<i32> {
    if !(quantized_multiplier >= 0 && (-31 <= shift && shift <= 30)) {
        return Err(BLiteError::InCompatibleCasting);
    }

    let total_shift = (31 - shift) as i64;
    let round = 1i64 << (total_shift - 1);
    let mut result = x as i64 * (quantized_multiplier as i64) + round;
    result = result >> total_shift;

    if !(core::i32::MIN as i64 <= result && result <= core::i32::MAX as i64) {
        return Err(BLiteError::InCompatibleCasting);
    }

    Ok(result as i32)
}

pub fn quantize(scale: f32, zero_point: i32, f: f32) -> Result<i32> {
    let tmp = (f / scale).round();

    // overflow check
    if !(core::i32::MIN as f32 <= tmp && tmp <= core::i32::MAX as f32) {
        return Err(BLiteError::FatalError);
    }

    let q = zero_point + tmp as i32;
    Ok(q)
}

pub fn dequantize(scale: f32, zero_point: i32, q: i8) -> Result<f32> {
    let f = scale * (q as i32 - zero_point) as f32;
    Ok(f)
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_get_quantized_convolutional_multiplier() {
        assert_eq!(1, 1)
    }

    #[test]
    fn test_quantize_multiplier() {
        let tt = [
            (0.0, (0, 0)),
            (0.1, (1717986918, -3)),
            (0.01, (1374389535, -6)),
            (0.03, (2061584302, -5)),
        ];
        for (real_multiplier, expected) in tt {
            assert_eq!(
                expected,
                super::quantize_multiplier(real_multiplier).unwrap()
            );
        }
    }

    #[test]
    fn multiply_by_quantized_multiplier() {
        let tt = [((1, 2, 30), 1), ((1, 1, 1), 0), ((1, 2, -30), 0)];
        for (real_multiplier, expected) in tt {
            let (x, quantized_multiplier, shift) = real_multiplier;
            assert_eq!(
                expected,
                super::multiply_by_quantized_multiplier(x, quantized_multiplier, shift).unwrap()
            );
        }
    }
}
