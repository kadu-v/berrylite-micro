use crate::builtin_op_data::BLiteOpParams;
use crate::kernel::micro_activation::get_activation;
use crate::kernel::micro_builtin_options::{
    BLiteBuiltinOption, BLiteBuiltinOption::*,
};
use crate::kernel::micro_builtin_options::BLiteBuiltinOption::Conv2DOptions;
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
pub struct Conv2D {}

impl Conv2D {
    const OPCODE: i32 = 3;

    pub fn conv2d<T: ArrayElem<T>>() -> BLiteOperator<T> {
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
            op.builtin_options_as_conv_2_doptions();
        let mut op_code = -1;
        let Some(builtin_option) = builtin_option else {
            return Err(NotFoundOption);
        };
        op_code = builtin_option
            .fused_activation_function()
            .0 as i32;
        let padding = builtin_option.padding().0 as usize;
        let activation = get_activation::<T>(op_code);
        let stride_w = builtin_option.stride_w();
        let stride_h = builtin_option.stride_h();
        let dilation_w_factor =
            builtin_option.dilation_w_factor();
        let dilation_h_factor =
            builtin_option.dilation_h_factor();

        let input_idx =
            op.inputs().unwrap().get(0) as usize;
        let input_h = tensors[input_idx].borrow().dims[1];
        let input_w = tensors[input_idx].borrow().dims[2];

        let filter_idx =
            op.inputs().unwrap().get(1) as usize;
        let filter_h = tensors[filter_idx].borrow().dims[1];
        let filter_w = tensors[filter_idx].borrow().dims[2];

        let output_idx =
            op.outputs().unwrap().get(0) as usize;
        let output_h = tensors[output_idx].borrow().dims[1];
        let output_w = tensors[output_idx].borrow().dims[2];

        let (
            padding_w,
            padding_w_offset,
            padding_h,
            padding_h_offset,
        ) = Self::compute_padding_height_width(
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
        Ok(BLiteBuiltinOption::Conv2DOptions {
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
        })
    }

    fn compute_padding_height_width(
        padding: usize,
        stride_h: i32,
        stride_w: i32,
        dilation_h_factor: i32,
        dilation_w_factor: i32,
        input_h: i32,
        input_w: i32,
        filter_h: i32,
        filter_w: i32,
        output_h: i32,
        output_w: i32,
    ) -> (i32, i32, i32, i32) {
        let out_height = Self::compute_out_size(
            padding,
            input_h,
            filter_h,
            stride_h,
            dilation_h_factor,
        );
        let out_width = Self::compute_out_size(
            padding,
            input_w,
            filter_w,
            stride_w,
            dilation_w_factor,
        );
        let (height, offset_h) =
            Self::compute_padding_with_offset(
                stride_h,
                dilation_h_factor,
                input_h,
                filter_h,
                out_height,
            );
        let (width, offset_w) =
            Self::compute_padding_with_offset(
                stride_w,
                dilation_w_factor,
                input_w,
                filter_w,
                out_width,
            );
        (height, offset_h, width, offset_w)
    }

    fn compute_out_size(
        padding: usize,
        image_size: i32,
        filter_size: i32,
        stride: i32,
        dilation_rate: i32,
    ) -> i32 {
        let effective_filter_size =
            (filter_size - 1) * dilation_rate + 1;

        if stride == 0 {
            return 0;
        }

        // padding 0: same, 1: valid
        if padding == 0 {
            (image_size + stride - 1) / stride
        } else if padding == 1 {
            (image_size + stride - effective_filter_size)
                / stride
        } else {
            0
        }
    }

    fn compute_padding_with_offset(
        stride: i32,
        dilation_rate: i32,
        input_size: i32,
        filter_size: i32,
        out_size: i32,
    ) -> (i32, i32) {
        let effective_filter_size =
            (filter_size - 1) * dilation_rate + 1;
        let mut total_padding =
            (out_size - 1) * stride * effective_filter_size
                - input_size;
        total_padding = if total_padding > 0 {
            total_padding
        } else {
            0
        };
        let offset = total_padding % 2;
        let pad = total_padding / 2;
        return (pad, offset);
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

        let idx_filter = node.inputs[1] as usize;
        let filter = tensors[idx_filter].borrow();
        let filter_height = filter.dims[1];
        let filter_width = filter.dims[2];
        let filter_input_depth = filter.dims[3];

        let idx_bias = node.inputs[2] as usize;
        let bias = tensors[idx_bias].borrow();

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output].borrow_mut();
        let output_height = output.dims[1];
        let output_width = output.dims[2];
        let output_depth = output.dims[3];

        // TODO: What is this?
        let batchs = input.dims[0]; // TODO: min(input.dims[0], output.dims[0])
        let groups = input_depth / filter_input_depth;
        let filters_per_group = output_depth / groups;

        let Conv2DOptions {
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
        } = builtin_option else {
            return Err(NotCompatibleOption)
        };

        for batch in 0..batchs {
            for out_y in 0..output_height {
                let in_y_origin =
                    (out_y * stride_h) - padding_h;
                for out_x in 0..output_width {
                    let in_x_origin =
                        (out_x * stride_w) - padding_w;
                    for out_channel in 0..output_depth {
                        let group =
                            out_channel / filters_per_group;
                        let mut total: T =
                            Default::default();
                        for filter_y in 0..filter_height {
                            let in_y = in_y_origin
                                + dilation_h_factor
                                    * filter_y;
                            for filter_x in 0..filter_width
                            {
                                let in_x = in_x_origin
                                    + dilation_w_factor
                                        * filter_x;
                                let is_point_inside_image =
                                    (in_x >= 0)
                                        && (in_x
                                            < input_width)
                                        && (in_y >= 0)
                                        && (in_y
                                            < input_height);
                                if !is_point_inside_image {
                                    continue;
                                }

                                for in_channel in
                                    0..filter_input_depth
                                {
                                    let input_v_idx = Self::offset(input_height, input_width, input_depth, batch, in_y, in_x, in_channel + group * filter_input_depth);
                                    let input_v = input
                                        .data
                                        [input_v_idx
                                            as usize];
                                    let filter_v_idx = Self::offset(filter_height, filter_width, filter_input_depth, out_channel, filter_y, filter_x, in_channel);
                                    let filter_v = filter
                                        .data
                                        [filter_v_idx
                                            as usize];
                                    total = total
                                        + (input_v
                                            * filter_v);
                                    let bias_v = bias.data
                                        [out_channel
                                            as usize];

                                    let output_v_idx =
                                        Self::offset(
                                            output_height,
                                            output_width,
                                            output_depth,
                                            batch,
                                            out_y,
                                            out_x,
                                            out_channel,
                                        );

                                    if let Some(
                                        activation,
                                    ) = activation
                                    {
                                        output.data[output_v_idx as usize] = activation(total + bias_v);
                                    } else {
                                        output.data
                                            [output_v_idx
                                                as usize] =
                                            total + bias_v;
                                    }
                                }
                            }
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
