use num_traits::{AsPrimitive, FromPrimitive};

use crate::kernel::micro_activation::get_activation;
use crate::kernel::micro_builtin_options::{BLiteBuiltinOption, BLiteBuiltinOption::*};
use crate::kernel::utils::calc_per_channel_multiplier_shift;
use crate::kernel::utils::padding::compute_padding_height_width;
use crate::kernel::utils::quantization_multiplier::{
    multiply_by_quantized_multiplier, quantize_multiplier,
};
use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::{ArrayElem, BLiteQuantizationParams};
use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteError::{self, *};
use crate::micro_erros::Result;
use crate::micro_node::BLiteNode;
use crate::micro_registration::BLiteRegistration;
use crate::micro_slice::alloc_array_mut;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::cmp::{max, min};
use core::fmt::Debug;

use crate::kernel::micro_operator::BLiteOperator;

#[derive(Debug, Clone, Copy)]
pub struct OpConv2DInt8 {}

impl OpConv2DInt8 {
    const OPCODE: i32 = 3;

    pub fn conv2d_int8<'a, T: ArrayElem<T>, S: ArenaAllocator>() -> BLiteOperator<'a, T, S> {
        BLiteOperator {
            registration: Self::registration(),
            parser: Self::parser,
        }
    }

    pub fn parser<'a, T: ArrayElem<T>>(
        allocator: &mut impl ArenaAllocator,
        op: Operator,
        tensors: &mut [BLiteTensor<'a, T>],
    ) -> Result<BLiteBuiltinOption<'a, T>> {
        let builtin_option = op.builtin_options_as_conv_2_doptions();
        let Some(builtin_option) = builtin_option else {
            return Err(NotFoundOption);
        };
        let op_code = builtin_option.fused_activation_function().0 as i32;
        let padding = builtin_option.padding().0 as usize;
        let activation = get_activation::<i32>(op_code);
        let stride_w = builtin_option.stride_w();
        let stride_h = builtin_option.stride_h();
        let dilation_w_factor = builtin_option.dilation_w_factor();
        let dilation_h_factor = builtin_option.dilation_h_factor();

        let input_idx = op.inputs().unwrap().get(0) as usize;
        let input_h = tensors[input_idx]._b_tensor()?.borrow().dims[1];
        let input_w = tensors[input_idx]._b_tensor()?.borrow().dims[2];
        let (input_scale, input_zero_point) = {
            let Some(BLiteQuantizationParams { scale, zero_point }) =
                tensors[input_idx]._b_tensor()?.borrow().quant_params
            else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale[0], zero_point[0] as i32)
        };

        let filter_idx = op.inputs().unwrap().get(1) as usize;
        let filter_h = tensors[filter_idx]._b_tensor()?.borrow().dims[1];
        let filter_w = tensors[filter_idx]._b_tensor()?.borrow().dims[2];
        let (filter_scales, filter_zero_point) = {
            let Some(BLiteQuantizationParams { scale, zero_point }) =
                tensors[filter_idx]._b_tensor()?.borrow().quant_params
            else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale, zero_point[0] as i32)
        };

        let output_idx = op.outputs().unwrap().get(0) as usize;
        let output_h = tensors[output_idx]._b_tensor()?.borrow().dims[1];
        let output_w = tensors[output_idx]._b_tensor()?.borrow().dims[2];
        let output_ch = tensors[output_idx]._b_tensor()?.borrow().dims[3];
        let (output_scale, output_zero_point) = {
            let Some(BLiteQuantizationParams { scale, zero_point }) =
                tensors[output_idx]._b_tensor()?.borrow().quant_params
            else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale[0], zero_point[0] as i32)
        };

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
        let per_channel_multiplier = unsafe { alloc_array_mut(allocator, output_ch as usize) }?;
        let per_channel_shift = unsafe { alloc_array_mut(allocator, output_ch as usize) }?;
        calc_per_channel_multiplier_shift(
            input_scale,
            filter_scales,
            output_scale,
            per_channel_multiplier,
            per_channel_shift,
        )?;

        Ok(BLiteBuiltinOption::QuantizedConv2DOptions {
            op_code,
            activation,
            padding,
            padding_w,
            padding_h,
            padding_w_offset,
            padding_h_offset,
            stride_w,
            stride_h,
            dilation_w_factor,
            dilation_h_factor,
            // for quantization parameters
            input_offset: -input_zero_point,
            filter_offset: -filter_zero_point,
            output_offset: output_zero_point,
            per_channel_multiplier,
            per_channel_shift,
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
        let filter_depth = filter.dims[3];

        // Why does a bias exist depsite of setting use_bias=False
        let idx_bias = node.inputs[2] as usize;
        let bias = tensors[idx_bias]._i32_tensor()?.borrow();

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output]._b_tensor()?.borrow_mut();
        let output_height = output.dims[1];
        let output_width = output.dims[2];
        let output_depth = output.dims[3];

        let batches = input.dims[0] as usize; // TODO: min(input.dims[0], output.dims[0])

        // TODO: This setting is needed for grouped convolutions
        let groups = input_depth / filter_depth;
        let filters_per_group = output_depth / groups;

        let QuantizedConv2DOptions {
            op_code: _,
            activation,
            padding: _,
            stride_w,
            stride_h,
            dilation_w_factor,
            dilation_h_factor,
            padding_w,
            padding_h,
            padding_w_offset: _,
            padding_h_offset: _,
            // for quantization
            input_offset,
            filter_offset,
            output_offset,
            per_channel_multiplier,
            per_channel_shift,
        } = builtin_option
        else {
            return Err(NotCompatibleOption);
        };

        Self::kernel(
            input.data,
            filter.data,
            bias.data,
            output.data,
            input_height,
            input_width,
            input_depth,
            filter_height,
            filter_width,
            filter_depth,
            output_height,
            output_width,
            output_depth,
            stride_w,
            stride_h,
            dilation_w_factor,
            dilation_h_factor,
            padding_w,
            padding_h,
            filters_per_group,
            // for quantization
            input_offset,
            filter_offset,
            output_offset,
            per_channel_multiplier,
            per_channel_shift,
            batches,
            activation,
        )
    }

    #[inline(always)]
    pub fn kernel<T: ArrayElem<T>>(
        input_data: &[T],
        filter_data: &[T],
        bias_data: &[i32],
        output_data: &mut [T],
        //
        input_height: i32,
        input_width: i32,
        input_depth: i32,
        filter_height: i32,
        filter_width: i32,
        filter_depth: i32,
        output_height: i32,
        output_width: i32,
        output_depth: i32,
        //
        stride_w: i32,
        stride_h: i32,
        dilation_w_factor: i32,
        dilation_h_factor: i32,
        padding_w: i32,
        padding_h: i32,
        filters_per_group: i32,
        // for quantization
        input_offset: i32,
        _filter_offset: i32,
        output_offset: i32,
        per_channel_multiplier: &[i32],
        per_channel_shift: &[i32],
        //
        batches: usize,
        activation: Option<fn(i32) -> i32>,
    ) -> Result<()> {
        for batch in 0..batches {
            for out_y in 0..output_height {
                let in_y_origin = (out_y * stride_h) - padding_h;
                for out_x in 0..output_width {
                    let in_x_origin = (out_x * stride_w) - padding_w;
                    for out_channel in 0..output_depth {
                        let group = out_channel / filters_per_group;
                        let mut total = 0;
                        for filter_y in 0..filter_height {
                            let in_y = in_y_origin + dilation_h_factor * filter_y;
                            for filter_x in 0..filter_width {
                                let in_x = in_x_origin + dilation_w_factor * filter_x;
                                let is_point_inside_image = (in_x >= 0)
                                    && (in_x < input_width)
                                    && (in_y >= 0)
                                    && (in_y < input_height);
                                if !is_point_inside_image {
                                    continue;
                                }

                                for in_channel in 0..filter_depth {
                                    let input_v_idx = Self::offset(
                                        input_height,
                                        input_width,
                                        input_depth,
                                        batch as i32,
                                        in_y,
                                        in_x,
                                        in_channel + group * filter_depth,
                                    );
                                    let input_v =
                                        AsPrimitive::<i8>::as_(input_data[input_v_idx as usize])
                                            as i32;
                                    let filter_v_idx = Self::offset(
                                        filter_height,
                                        filter_width,
                                        filter_depth,
                                        out_channel,
                                        filter_y,
                                        filter_x,
                                        in_channel,
                                    );
                                    let filter_v =
                                        AsPrimitive::<i8>::as_(filter_data[filter_v_idx as usize])
                                            as i32;
                                    // All filter_offset is 0 by the implementation of tensorflow lite
                                    total += filter_v * (input_v + input_offset);
                                }
                            }
                        }
                        let bias_v = bias_data[out_channel as usize];

                        total += bias_v;

                        total = multiply_by_quantized_multiplier(
                            total,
                            per_channel_multiplier[out_channel as usize],
                            per_channel_shift[out_channel as usize],
                        )?;

                        if let Some(activation) = activation {
                            total = activation(total);
                        }
                        total += output_offset;
                        total = max(total, core::i8::MIN as i32);
                        total = min(total, core::i8::MAX as i32);

                        let output_v_idx = Self::offset(
                            output_height,
                            output_width,
                            output_depth,
                            batch as i32,
                            out_y,
                            out_x,
                            out_channel,
                        );

                        output_data[output_v_idx as usize] =
                            FromPrimitive::from_i8(total as i8).unwrap();
                    }
                }
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn offset(h: i32, w: i32, d: i32, i0: i32, i1: i32, i2: i32, i3: i32) -> i32 {
        ((i0 * h + i1) * w + i2) * d + i3
    }
}
