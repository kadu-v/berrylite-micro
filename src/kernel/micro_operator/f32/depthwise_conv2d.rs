use crate::kernel::micro_activation::get_activation;
use crate::kernel::micro_builtin_options::{BLiteBuiltinOption, BLiteBuiltinOption::*};
use crate::kernel::utils::padding::compute_padding_height_width;
use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::ArrayElem;
use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteError::*;
use crate::micro_erros::Result;
use crate::micro_node::BLiteNode;
use crate::micro_registration::BLiteRegistration;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::fmt::Debug;

use crate::kernel::micro_operator::BLiteOperator;

#[derive(Debug, Clone, Copy)]
pub struct OpDepthWiseConv2D {}

impl OpDepthWiseConv2D {
    const OPCODE: i32 = 4;

    pub fn depthwise_conv2d<'a, T: ArrayElem<T>, S: ArenaAllocator>() -> BLiteOperator<'a, T, S> {
        BLiteOperator {
            registration: Self::registration(),
            parser: Self::parser,
        }
    }

    pub fn parser<'a, T: ArrayElem<T>>(
        _allocator: &mut impl ArenaAllocator,
        op: Operator,
        tensors: &mut [BLiteTensor<'a, T>],
    ) -> Result<BLiteBuiltinOption<'a, T>> {
        let builtin_option = op.builtin_options_as_depthwise_conv_2_doptions();
        let Some(builtin_option) = builtin_option else {
            return Err(NotFoundOption);
        };
        let op_code = builtin_option.fused_activation_function().0 as i32;

        let padding = builtin_option.padding().0 as usize;
        let activation = get_activation::<T>(op_code);
        let stride_w = builtin_option.stride_w();
        let stride_h = builtin_option.stride_h();
        let depth_multiplier = builtin_option.depth_multiplier();
        let dilation_w_factor = builtin_option.dilation_w_factor();
        let dilation_h_factor = builtin_option.dilation_h_factor();

        let input_idx = op.inputs().unwrap().get(0) as usize;
        let input_h = tensors[input_idx]._b_tensor()?.borrow().dims[1];
        let input_w = tensors[input_idx]._b_tensor()?.borrow().dims[2];

        let filter_idx = op.inputs().unwrap().get(1) as usize;
        let filter_h = tensors[filter_idx]._b_tensor()?.borrow().dims[1];
        let filter_w = tensors[filter_idx]._b_tensor()?.borrow().dims[2];

        let output_idx = op.outputs().unwrap().get(0) as usize;
        let output_h = tensors[output_idx]._b_tensor()?.borrow().dims[1];
        let output_w = tensors[output_idx]._b_tensor()?.borrow().dims[2];

        let (padding_w, padding_w_offset, padding_h, padding_h_offset) =
            compute_padding_height_width(
                padding,
                stride_h,
                stride_w,
                dilation_h_factor,
                dilation_w_factor,
                input_h,
                input_w,
                filter_h,
                filter_w,
                output_h,
                output_w,
            );
        Ok(BLiteBuiltinOption::DepthWiseConv2DOptions {
            op_code,
            activation,
            padding,
            padding_w,
            padding_h,
            padding_w_offset,
            padding_h_offset,
            stride_w,
            stride_h,
            depth_multiplier,
            dilation_w_factor,
            dilation_h_factor,
        })
    }

    pub fn registration<'a, T: ArrayElem<T>>() -> BLiteRegistration<'a, T> {
        BLiteRegistration::new(Self::OPCODE, Self::eval::<T>, NotInitialize)
    }

    pub fn eval<'a, T: ArrayElem<T>>(
        _context: &BLiteContext,
        tensors: &'a mut [BLiteTensor<'a, T>],
        node: &BLiteNode<'a>,
        builtin_option: BLiteBuiltinOption<T>,
    ) -> Result<()> {
        let idx_input = node.inputs[0] as usize;
        let input = tensors[idx_input]._b_tensor()?.borrow();
        let input_height = input.dims[1];
        let input_width = input.dims[2];
        let input_depth = input.dims[3];

        let idx_filter = node.inputs[1] as usize;
        let filter = tensors[idx_filter]._b_tensor()?.borrow();
        let filter_height = filter.dims[1];
        let filter_width = filter.dims[2];
        let filter_input_depth = filter.dims[3];

        let idx_bias = node.inputs[2] as usize;
        let bias = tensors[idx_bias]._b_tensor()?.borrow();

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output]._b_tensor()?.borrow_mut();
        let output_height = output.dims[1];
        let output_width = output.dims[2];
        let output_depth = output.dims[3];

        // TODO: What is this?
        let batchs = input.dims[0]; // TODO: min(input.dims[0], output.dims[0])

        let DepthWiseConv2DOptions {
            op_code:_,
            activation,
            padding:_,
            stride_w,
            stride_h,
            dilation_w_factor,
            dilation_h_factor,
            padding_w,
            padding_h,
            padding_w_offset:_,
            padding_h_offset:_ ,
            depth_multiplier,
        } = builtin_option else {
            return Err(NotCompatibleOption);
        };

        for batch in 0..batchs {
            for out_y in 0..output_height {
                for out_x in 0..output_width {
                    for in_channel in 0..input_depth {
                        for m in 0..depth_multiplier {
                            let out_channel = m + in_channel * depth_multiplier;
                            let in_x_origin = (out_x * stride_w) - padding_w;
                            let in_y_origin = (out_y * stride_h) - padding_h;
                            let mut total: T = Default::default();
                            for filter_y in 0..filter_height {
                                for filter_x in 0..filter_width {
                                    let in_x = in_x_origin + dilation_w_factor * filter_x;
                                    let in_y = in_y_origin + dilation_h_factor * filter_y;
                                    let is_point_inside_image = (in_x >= 0)
                                        && (in_x < input_width)
                                        && (in_y >= 0)
                                        && (in_y < input_height);
                                    if is_point_inside_image {
                                        let input_v_idx = Self::offset(
                                            input_height,
                                            input_width,
                                            input_depth,
                                            batch,
                                            in_y,
                                            in_x,
                                            in_channel,
                                        );
                                        let input_v = input.data[input_v_idx as usize];
                                        let filter_v_idx = Self::offset(
                                            filter_height,
                                            filter_width,
                                            filter_input_depth,
                                            0,
                                            filter_y,
                                            filter_x,
                                            out_channel,
                                        );
                                        let filter_v = filter.data[filter_v_idx as usize];
                                        total += input_v * filter_v;
                                    }
                                }
                            }
                            let bias_v = bias.data[out_channel as usize];
                            let output_v_idx = Self::offset(
                                output_height,
                                output_width,
                                output_depth,
                                batch,
                                out_y,
                                out_x,
                                out_channel,
                            );

                            if let Some(activation) = activation {
                                output.data[output_v_idx as usize] = activation(total + bias_v);
                            } else {
                                output.data[output_v_idx as usize] = total + bias_v;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn offset(h: i32, w: i32, d: i32, i0: i32, i1: i32, i2: i32, i3: i32) -> i32 {
        ((i0 * h + i1) * w + i2) * d + i3
    }
}
