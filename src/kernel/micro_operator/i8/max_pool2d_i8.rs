use num_traits::{AsPrimitive, FromPrimitive};

use crate::kernel::micro_activation::{calculate_fused_activation_range_quantized, get_activation};
use crate::kernel::micro_builtin_options::{
    BLiteBuiltinOption,
    BLiteBuiltinOption::{NotInitialize, QuantizedMaxPool2DOptions},
};
use crate::kernel::utils::padding::compute_padding_height_width;
use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::{ArrayElem, BLiteQuantizationParams};
use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteError::{self, *};
use crate::micro_erros::Result;
use crate::micro_node::BLiteNode;
use crate::micro_registration::BLiteRegistration;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::fmt::Debug;

use crate::kernel::micro_operator::BLiteOperator;

#[derive(Debug, Clone, Copy)]
pub struct OpMaxPool2DInt8 {}

impl OpMaxPool2DInt8 {
    const OPCODE: i32 = 17;

    pub fn max_pool2d_int8<'a, T: ArrayElem<T>, S: ArenaAllocator>() -> BLiteOperator<'a, T, S> {
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
        let builtin_option = op.builtin_options_as_pool_2_doptions();
        let Some(builtin_option) = builtin_option else {
            return Err(NotFoundOption);
        };
        let op_code = builtin_option.fused_activation_function().0 as i32;

        let padding = builtin_option.padding().0 as usize;
        let stride_w = builtin_option.stride_w();
        let stride_h = builtin_option.stride_h();
        let filter_w = builtin_option.filter_width();
        let filter_h = builtin_option.filter_height();

        let input_idx = op.inputs().unwrap().get(0) as usize;
        let input_h = tensors[input_idx]._b_tensor()?.borrow().dims[1];
        let input_w = tensors[input_idx]._b_tensor()?.borrow().dims[2];

        let output_idx = op.outputs().unwrap().get(0) as usize;
        let output_h = tensors[output_idx]._b_tensor()?.borrow().dims[1];
        let output_w = tensors[output_idx]._b_tensor()?.borrow().dims[2];
        let (output_scale, output_zero_point) = {
            let Some(BLiteQuantizationParams { scale, zero_point }) =
                tensors[output_idx]._b_tensor()?.borrow().quant_params
            else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale[0], zero_point[0] as i32)
        };
        let (fused_activation_min, fused_activation_max) =
            calculate_fused_activation_range_quantized(output_scale, output_zero_point, op_code)?;

        let (padding_w, padding_w_offset, padding_h, padding_h_offset) =
            compute_padding_height_width(
                padding, stride_h, stride_w, /* dilation_h_factor */ 1,
                /*dilation_w_factor */ 1, input_h, input_w, filter_h, filter_w, output_h,
                output_w,
            );
        Ok(BLiteBuiltinOption::QuantizedMaxPool2DOptions {
            op_code,
            fused_activation_min,
            fused_activation_max,
            padding,
            stride_w,
            stride_h,
            filter_w,
            filter_h,
            padding_w,
            padding_h,
            padding_w_offset,
            padding_h_offset,
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

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output]._b_tensor()?.borrow_mut();
        let output_height = output.dims[1];
        let output_width = output.dims[2];
        let output_depth = output.dims[3];

        let batches = input.dims[0]; // TODO: min(input.dims[0], output.dims[0])
        let QuantizedMaxPool2DOptions {
            op_code: _,
            fused_activation_min,
            fused_activation_max,
            padding: _,
            stride_w,
            stride_h,
            filter_w,
            filter_h,
            padding_w,
            padding_h,
            padding_w_offset: _,
            padding_h_offset: _,
        } = builtin_option
        else {
            return Err(NotCompatibleOption);
        };
        Self::kernel(
            input.data,
            output.data,
            input_height,
            input_width,
            input_depth,
            output_height,
            output_width,
            output_depth,
            stride_w,
            stride_h,
            filter_w,
            filter_h,
            padding_w,
            padding_h,
            batches as usize,
            fused_activation_min,
            fused_activation_max,
        )
    }

    pub fn kernel<T: ArrayElem<T>>(
        input_data: &[T],
        output_data: &mut [T],
        //
        input_height: i32,
        input_width: i32,
        input_depth: i32,
        output_height: i32,
        output_width: i32,
        output_depth: i32,
        //
        stride_w: i32,
        stride_h: i32,
        filter_w: i32,
        filter_h: i32,
        padding_w: i32,
        padding_h: i32,
        //
        batches: usize,
        fused_activation_min: i32,
        fused_activation_max: i32,
    ) -> Result<()> {
        for batch in 0..batches {
            for out_y in 0..output_height {
                for out_x in 0..output_width {
                    for channel in 0..output_depth {
                        let in_x_origin = (out_x * stride_w) - padding_w;
                        let in_y_origin = (out_y * stride_h) - padding_h;
                        let filter_x_start = core::cmp::max(0, -in_x_origin);
                        let filter_x_end = core::cmp::min(filter_w, input_width - in_x_origin);
                        let filter_y_start = core::cmp::max(0, -in_y_origin);
                        let filter_y_end = core::cmp::min(filter_h, input_height - in_y_origin);
                        let mut max = FromPrimitive::from_i8(core::i8::MIN).unwrap();
                        for filter_y in filter_y_start..filter_y_end {
                            for filter_x in filter_x_start..filter_x_end {
                                let in_y = in_y_origin + filter_y;
                                let in_x = in_x_origin + filter_x;
                                let input_v_idx = Self::offset(
                                    input_height,
                                    input_width,
                                    input_depth,
                                    batch as i32,
                                    in_y,
                                    in_x,
                                    channel,
                                );
                                let input_v = input_data[input_v_idx as usize];
                                if input_v > max {
                                    max = input_v;
                                }
                            }
                        }

                        let mut max = AsPrimitive::<i8>::as_(max) as i32;
                        max = core::cmp::max(max, fused_activation_min);
                        max = core::cmp::min(max, fused_activation_max);

                        let output_v_idx = Self::offset(
                            output_height,
                            output_width,
                            output_depth,
                            batch as i32,
                            out_y,
                            out_x,
                            channel,
                        );
                        output_data[output_v_idx as usize] =
                            FromPrimitive::from_i8(max as i8).unwrap();
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
