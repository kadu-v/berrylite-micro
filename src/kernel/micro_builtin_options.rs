use std::fmt::Debug;

use crate::micro_array::ArrayElem;

#[derive(Debug, Clone, Copy)]
pub enum BLiteBuiltinOption<T: Debug + ArrayElem<T>> {
    FullyConnectedOptions {
        op_code: i32,
        activation: Option<fn(T) -> T>,
    },
    /// input/filter/output_offset are negative values of input/filter/output_zero_point
    QuantizedFullyConnectedOptions {
        op_code: i32,
        activation: Option<fn(i32) -> i32>,
        input_offset: i32,
        filter_offset: i32,
        output_offset: i32,
        output_multiplier: i32,
        output_shift: i32,
        // output_activation_min: i32,
        // outpu_activation_max: i32,
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
    NotInitialize,
}
