use crate::kernel::micro_activation::get_activation;
use crate::kernel::micro_builtin_options::{
    BLiteBuiltinOption,
    BLiteBuiltinOption::{NotInitialize, MaxPool2DOptions},
};
use crate::kernel::micro_operator::padding::compute_padding_height_width;
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
pub struct MaxPool2D {}

impl MaxPool2D {
    const OPCODE: i32 = 17;

    pub fn max_pool2d<T: ArrayElem<T>>() -> BLiteOperator<T>
    {
        BLiteOperator {
            registration: Self::registration(),
            parser: Self::parser,
        }
    }

    pub fn parser<T: ArrayElem<T>>(
        op: Operator,
        tensors: &mut [BLiteTensor<'_, T>],
    ) -> Result<BLiteBuiltinOption<T>> {
        let builtin_option =
            op.builtin_options_as_pool_2_doptions();
        let Some(builtin_option) = builtin_option else {
            return Err(NotFoundOption);
        };
        let op_code = builtin_option
            .fused_activation_function()
            .0 as i32;
        let activation = get_activation::<T>(op_code);
        let padding = builtin_option.padding().0 as usize;
        let stride_w = builtin_option.stride_w();
        let stride_h = builtin_option.stride_h();
        let filter_w = builtin_option.filter_width();
        let filter_h = builtin_option.filter_height();

        let input_idx =
            op.inputs().unwrap().get(0) as usize;
        let input_h = tensors[input_idx].borrow().dims[1];
        let input_w = tensors[input_idx].borrow().dims[2];

        let output_idx =
            op.outputs().unwrap().get(0) as usize;
        let output_h = tensors[output_idx].borrow().dims[1];
        let output_w = tensors[output_idx].borrow().dims[2];

        let (
            padding_w,
            padding_w_offset,
            padding_h,
            padding_h_offset,
        ) = compute_padding_height_width(
            padding,
            stride_h,
            stride_w,
            /* dilation_h_factor */ 1,
            /*dilation_w_factor */ 1,
            input_h,
            input_w,
            filter_h,
            filter_w,
            output_h,
            output_w,
        );
        Ok(BLiteBuiltinOption::MaxPool2DOptions {
            op_code,
            activation,
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

    pub fn registration<T: ArrayElem<T>>(
    ) -> BLiteRegistration<T> {
        BLiteRegistration::new(
            Self::OPCODE,
            Self::eval::<T>,
            NotInitialize,
        )
    }

    pub fn eval<'a, T: ArrayElem<T>>(
        _context: &BLiteContext<'a, T>,
        tensors: &'a mut [BLiteTensor<'a, T>],
        node: &BLiteNode<'a>,
        builtin_option: BLiteBuiltinOption<T>,
    ) -> Result<()> {
        let idx_input = node.inputs[0] as usize;
        let input = tensors[idx_input].borrow();
        let input_height = input.dims[1];
        let input_width = input.dims[2];
        let input_depth = input.dims[3];

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output].borrow_mut();
        let output_height = output.dims[1];
        let output_width = output.dims[2];
        let output_depth = output.dims[3];

        let batchs = input.dims[0]; // TODO: min(input.dims[0], output.dims[0])
        let MaxPool2DOptions { 
            op_code:_,
            activation,
            padding:_,
            stride_w,
            stride_h,
            filter_w, 
            filter_h,
            padding_w,
            padding_h,
            padding_w_offset:_,
            padding_h_offset:_, 
        } = builtin_option else {
            return Err(NotCompatibleOption)
        };

        for batch in 0..batchs {
            for out_y in 0..output_height {
                for out_x in 0..output_width {
                    for channel in 0..output_depth {
                        let in_x_origin = (out_x * stride_w) - padding_w;
                        let in_y_origin = (out_y * stride_h) - padding_h;
                        let filter_x_start = core::cmp::max(0, -in_x_origin);
                        let filter_x_end = core::cmp::min(filter_w, input_width - in_x_origin);
                        let filter_y_start = core::cmp::max(0, -in_y_origin);
                        let filter_y_end = core::cmp::min(filter_h, input_height - in_y_origin);
                        let mut max = Default::default();
                        for filter_y in filter_y_start..filter_y_end {
                            for filter_x in filter_x_start..filter_x_end {
                                let in_y = in_y_origin + filter_y;
                                let in_x = in_x_origin + filter_x;
                                let input_v_idx = Self::offset(input_height, input_width, input_depth, batch, in_y, in_x, channel);
                                let input_v = input
                                    .data
                                    [input_v_idx
                                        as usize];
                                if input_v > max {
                                    max = input_v;
                                }
                            }
                        }
                        let output_v_idx =
                        Self::offset(
                            output_height,
                            output_width,
                            output_depth,
                            batch,
                            out_y,
                            out_x,
                            channel,
                        );
                        if let Some(
                            activation,
                        ) = activation
                        {
                            output.data[output_v_idx as usize] = activation(max);
                        } else {
                            output.data
                                [output_v_idx
                                    as usize] =
                                max;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn offset(
        h: i32,
        w: i32,
        d: i32,
        i0: i32,
        i1: i32,
        i2: i32,
        i3: i32,
    ) -> i32 {
        ((i0 * h + i1) * w + i2) * d + i3
    }
}
