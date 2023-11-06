pub mod fixed_point;
pub mod padding;
pub mod quantization;
pub mod types;

#[cfg(test)]
pub mod testing;

use crate::micro_errors::Result;
use quantization::quantize_multiplier;

pub fn calc_per_channel_multiplier_shift(
    input_scale: f32,
    filter_scales: &[f32],
    output_scale: f32,
    per_channel_multiplier: &mut [i32],
    per_channel_shift: &mut [i32],
) -> Result<()> {
    for (i, &filter_scale) in filter_scales.iter().enumerate() {
        let effective_output_scale = input_scale as f64 * filter_scale as f64 / output_scale as f64;
        let (multiplier, shift) = quantize_multiplier(effective_output_scale)?;
        per_channel_multiplier[i] = multiplier;
        per_channel_shift[i] = shift;
    }
    Ok(())
}
