use core::fmt::Debug;

use crate::micro_array::ArrayElem;

#[derive(Debug, Clone, Copy)]
pub enum BLiteBuiltinOption<T: Debug + ArrayElem<T>> {
    FullyConnectedOptions {
        op_code: i32,
        activation: Option<fn(T) -> T>,
    },
    Conv2DOptions {
        op_code: i32,
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
