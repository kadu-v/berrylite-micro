use num_traits::{AsPrimitive, FromPrimitive};

use crate::kernel;
use crate::kernel::kernel_utils::{
    get_quantized_convolutional_multiplier, multiply_by_quantized_multiplier, quantize_multiplier,
};
use crate::kernel::micro_activation::get_activation;
use crate::kernel::micro_builtin_options::{BLiteBuiltinOption, BLiteBuiltinOption::*};
use crate::micro_array::{ArrayElem, BLiteQuantizationParams};
use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteError::{self, *};
use crate::micro_erros::Result;
use crate::micro_node::BLiteNode;
use crate::micro_registration::BLiteRegistration;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::cmp::{max, min};
use core::fmt::Debug;
use std::borrow::BorrowMut;

use crate::kernel::micro_operator::BLiteOperator;

#[derive(Debug, Clone, Copy)]
pub struct OpFullyConnectedInt8 {}

impl OpFullyConnectedInt8 {
    const OPCODE: i32 = 9;

    pub fn fully_connected_int8<T: ArrayElem<T>>() -> BLiteOperator<T> {
        BLiteOperator {
            registration: Self::registration(),
            parser: Self::parser,
        }
    }

    pub fn parser<T: ArrayElem<T>>(
        op: Operator,
        tensors: &mut [BLiteTensor<'_, T>],
    ) -> Result<BLiteBuiltinOption<T>> {
        let builtin_option = op.builtin_options_as_fully_connected_options();
        let mut op_code = -1;
        if let Some(builtin_option) = builtin_option {
            op_code = builtin_option.fused_activation_function().0 as i32;
        }

        let input_idx = op.inputs().unwrap().get(0) as usize;
        let (input_scale, input_zero_point) = {
            let Some(BLiteQuantizationParams {
               scale, zero_point
            }) = tensors[input_idx].borrow().quant_params else {
                return Err(BLiteError::FailedToAllocateMemory);
            };
            (scale, zero_point)
        };

        let filter_idx = op.inputs().unwrap().get(1) as usize;
        let (filter_scale, filter_zero_point) = {
            let Some(BLiteQuantizationParams {
               scale, zero_point
            }) = tensors[filter_idx].borrow().quant_params else {
                return Err(BLiteError::FailedToAllocateMemory);
            };
            (scale, zero_point)
        };

        let bias_idx = op.inputs().unwrap().get(2);
        let bias_scale = if bias_idx >= 0 {
            let Some(BLiteQuantizationParams {
                scale, ..
             }) = tensors[bias_idx as usize].borrow().quant_params else {
                 return Err(BLiteError::FailedToAllocateMemory);
             };
            Some(scale)
        } else {
            None
        };

        let output_idx = op.outputs().unwrap().get(0) as usize;
        let (output_scale, output_zero_point) = {
            let Some(BLiteQuantizationParams {
               scale, zero_point
            }) = tensors[output_idx].borrow().quant_params else {
                return Err(BLiteError::FailedToAllocateMemory);
            };
            (scale, zero_point)
        };

        let real_multiplier = get_quantized_convolutional_multiplier(
            input_scale,
            filter_scale,
            output_scale,
            bias_scale,
        )?;
        let (output_multiplier, output_shift) = quantize_multiplier(real_multiplier)?;

        let activation = get_activation::<T>(op_code);

        Ok(BLiteBuiltinOption::QuantizedFullyConnectedOptions {
            op_code,
            activation: activation,
            input_offset: -input_zero_point,
            filter_offset: -filter_zero_point,
            output_offset: output_zero_point,
            output_multiplier,
            output_shift,
        })
    }

    pub fn registration<T: ArrayElem<T>>() -> BLiteRegistration<T> {
        BLiteRegistration::new(Self::OPCODE, Self::eval::<T>, NotInitialize)
    }

    pub fn eval<'a, T: ArrayElem<T>>(
        _context: &BLiteContext<'a, T>,
        tensors: &'a mut [BLiteTensor<'a, T>],
        node: &BLiteNode<'a>,
        builtin_option: BLiteBuiltinOption<T>,
    ) -> Result<()> {
        let QuantizedFullyConnectedOptions {
            op_code,
            activation,
            input_offset,
            filter_offset,
            output_offset,
            output_multiplier,
            output_shift,
        } = builtin_option else {
            return Err(NotInitializeActivation);
        };

        let idx_input = node.inputs[0] as usize;
        let input = tensors[idx_input].borrow();

        let idx_filter = node.inputs[1] as usize;
        let filter = tensors[idx_filter].borrow();

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output].borrow_mut();

        let idx_bias = node.inputs[2];

        // println!("[Input]: {:?}", &input);
        // println!("[Filter]: {:?}", &filter);
        // println!("[Bias]: {:?}", &bias.len());
        let batches = 1usize;
        let output_depth = filter.dims[filter.dims.len() - 2] as usize;
        let accum_depth = filter.dims[filter.dims.len() - 1] as usize;

        if idx_bias >= 0 {
            let bias = tensors[idx_bias as usize].borrow();

            Self::kernel(
                input.data,
                filter.data,
                Some(bias.data.as_ref()),
                output.data,
                input_offset,
                filter_offset,
                output_offset,
                output_depth,
                output_multiplier,
                output_shift,
                accum_depth,
                batches,
                activation,
            )
        } else {
            Self::kernel(
                input.data,
                filter.data,
                None,
                output.data,
                input_offset,
                filter_offset,
                output_offset,
                output_depth,
                output_multiplier,
                output_shift,
                accum_depth,
                batches,
                activation,
            )
        }
    }

    fn kernel<'a, T: ArrayElem<T>>(
        input_data: &[T],
        filter_data: &[T],
        bias_data: Option<&[T]>,
        output_data: &mut [T],
        input_offset: i32,
        filter_offset: i32,
        output_offset: i32,
        output_depth: usize,
        output_multiplier: i32,
        output_shift: i32,
        accum_depth: usize,
        batches: usize,
        activation: Option<fn(T) -> T>,
    ) -> Result<()> {
        // TODO:

        for batch in 0usize..batches {
            for out_d in 0usize..output_depth {
                let mut total = 0;
                for acc_d in 0usize..accum_depth {
                    let input_val =
                        AsPrimitive::<u8>::as_(input_data[batch * accum_depth + acc_d]) as i32;
                    let filter_val =
                        AsPrimitive::<u8>::as_(filter_data[out_d * accum_depth + acc_d]) as i32;
                    total += (input_val + input_offset) * (filter_val + filter_offset);
                }

                if let Some(bias_data) = bias_data {
                    total += AsPrimitive::<u8>::as_(bias_data[out_d]) as i32;
                }

                total = multiply_by_quantized_multiplier(total, output_multiplier, output_shift)?;
                total += output_offset;
                dbg!(total);
                total = max(total, core::i8::MIN as i32);
                total = min(total, core::i8::MAX as i32);
                // TODO: check the output value is included between the min and max of an output activation
                output_data[batch * output_depth + out_d] =
                    FromPrimitive::from_u8(total as u8).unwrap();
                // dbg!(total, output.data[batch * output_depth + out_d]);
            }
        }
        if let Some(activation) = activation {
            for i in 0..output_data.len() {
                output_data[i] = activation(output_data[i]);
            }
        }
        Ok(())
    }
}
