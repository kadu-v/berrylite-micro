use std::fmt::Debug;

use crate::micro_array::ArrayElem;

#[derive(Debug, Clone, Copy)]
pub enum BLiteBuiltinOption<'a, T: Debug + ArrayElem<T>> {
    FullyConnectedOptions {
        op_code: i32,
        activation: Option<fn(T) -> T>,
    },
    Conv2DOptions {
        op_code: i32, // activation operator code
        activation: Option<fn(T) -> T>,
        padding: usize, // 0: same, 1: valid
        padding_w: i32,
        padding_h: i32,
        padding_w_offset: i32,
        padding_h_offset: i32,
        stride_w: i32,
        stride_h: i32,
        dilation_w_factor: i32,
        dilation_h_factor: i32,
    },
    DepthWiseConv2DOptions {
        op_code: i32,
        activation: Option<fn(T) -> T>,
        padding: usize, // 0: same, 1: valid
        padding_w: i32,
        padding_h: i32,
        padding_w_offset: i32,
        padding_h_offset: i32,
        stride_w: i32,
        stride_h: i32,
        depth_multiplier: i32,
        dilation_w_factor: i32,
        dilation_h_factor: i32,
    },
    MaxPool2DOptions {
        op_code: i32,
        activation: Option<fn(T) -> T>,
        padding: usize, // 0: same, 1: valid
        padding_w: i32,
        padding_h: i32,
        padding_w_offset: i32,
        padding_h_offset: i32,
        stride_w: i32,
        stride_h: i32,
        filter_w: i32,
        filter_h: i32,
    },
    ReshapeOptions {},
    SoftMaxOptions {
        beta: f32,
    },
    /// input/filter/output_offset are negative values of input/filter/output_zero_point
    QuantizedFullyConnectedOptions {
        op_code: i32,
        fused_activation_min: i32,
        fused_activation_max: i32,
        // for quantization parameters
        input_offset: i32,
        filter_offset: i32,
        output_offset: i32,
        output_multiplier: i32,
        output_shift: i32,
        // output_activation_min: i32,
        // outpu_activation_max: i32,
    },
    QuantizedConv2DOptions {
        op_code: i32, // activation operator code
        fused_activation_min: i32,
        fused_activation_max: i32,
        padding: usize, // 0: same, 1: valid
        padding_w: i32,
        padding_h: i32,
        padding_w_offset: i32,
        padding_h_offset: i32,
        stride_w: i32,
        stride_h: i32,
        dilation_w_factor: i32,
        dilation_h_factor: i32,
        // for quantization parameters
        input_offset: i32,
        filter_offset: i32,
        output_offset: i32,
        per_channel_multiplier: &'a [i32],
        per_channel_shift: &'a [i32],
    },
    QuantizedDepthWiseConv2DOptions {
        op_code: i32,
        fused_activation_min: i32,
        fused_activation_max: i32,
        padding: usize, // 0: same, 1: valid
        padding_w: i32,
        padding_h: i32,
        padding_w_offset: i32,
        padding_h_offset: i32,
        stride_w: i32,
        stride_h: i32,
        depth_multiplier: i32,
        dilation_w_factor: i32,
        dilation_h_factor: i32,
        // for quantization parameters
        input_offset: i32,
        filter_offset: i32,
        output_offset: i32,
        per_channel_multiplier: &'a [i32],
        per_channel_shift: &'a [i32],
    },
    QuantizedMaxPool2DOptions {
        op_code: i32,
        fused_activation_min: i32,
        fused_activation_max: i32,
        padding: usize, // 0: same, 1: valid
        padding_w: i32,
        padding_h: i32,
        padding_w_offset: i32,
        padding_h_offset: i32,
        stride_w: i32,
        stride_h: i32,
        filter_w: i32,
        filter_h: i32,
    },
    QuantizedAvgPool2DOptions {
        op_code: i32,
        fused_activation_min: i32,
        fused_activation_max: i32,
        padding: usize, // 0: same, 1: valid
        padding_w: i32,
        padding_h: i32,
        padding_w_offset: i32,
        padding_h_offset: i32,
        stride_w: i32,
        stride_h: i32,
        filter_w: i32,
        filter_h: i32,
    },
    QuantizedReshapeOptions {},
    QuantizedSoftMaxOptions {
        beta: f32,
        input_scale: f32,
        input_zero_point: i32,
        output_scale: f32,
        output_zero_point: i32,
    },
    NotInitialize,
}
