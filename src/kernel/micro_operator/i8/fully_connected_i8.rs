use num_traits::{AsPrimitive, FromPrimitive};

use crate::kernel::micro_activation::get_activation;
use crate::kernel::micro_builtin_options::{BLiteBuiltinOption, BLiteBuiltinOption::*};
use crate::kernel::utils::quantization_multiplier::{
    get_quantized_convolution_multiplier, multiply_by_quantized_multiplier, quantize_multiplier,
};
use crate::micro_allocator::ArenaAllocator;
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

use crate::kernel::micro_operator::BLiteOperator;

#[derive(Debug, Clone, Copy)]
pub struct OpFullyConnectedInt8 {}

impl OpFullyConnectedInt8 {
    const OPCODE: i32 = 9;

    pub fn fully_connected_int8<'a, T: ArrayElem<T>, S: ArenaAllocator>() -> BLiteOperator<'a, T, S>
    {
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
        let builtin_option = op.builtin_options_as_fully_connected_options();
        let mut op_code = -1;
        if let Some(builtin_option) = builtin_option {
            op_code = builtin_option.fused_activation_function().0 as i32;
        }

        let input_idx = op.inputs().unwrap().get(0) as usize;
        let (input_scale, input_zero_point) = {
            let Some(BLiteQuantizationParams {
               scale, zero_point
            }) = tensors[input_idx]._b_tensor()?.borrow().quant_params else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale[0], zero_point[0] as i32)
        };

        let filter_idx = op.inputs().unwrap().get(1) as usize;
        let (filter_scale, filter_zero_point) = {
            let Some(BLiteQuantizationParams {
               scale, zero_point
            }) = tensors[filter_idx]._b_tensor()?.borrow().quant_params else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale[0], zero_point[0] as i32)
        };

        let bias_idx = op.inputs().unwrap().get(2);
        let bias_scale = if bias_idx >= 0 {
            let Some(BLiteQuantizationParams {
                scale, ..
             }) = tensors[bias_idx as usize]._i32_tensor()?.borrow().quant_params else {
                 return Err(BLiteError::NotFoundQuantParams);
             };
            Some(scale[0])
        } else {
            None
        };

        let output_idx = op.outputs().unwrap().get(0) as usize;
        let (output_scale, output_zero_point) = {
            let Some(BLiteQuantizationParams {
               scale, zero_point
            }) = tensors[output_idx]._b_tensor()?.borrow().quant_params else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale[0], zero_point[0] as i32)
        };

        //This computations is corresponded to CalculateOpDataFullyConnected
        let real_multiplier = get_quantized_convolution_multiplier(
            input_scale,
            filter_scale,
            output_scale,
            bias_scale,
        )?;
        let (output_multiplier, output_shift) = quantize_multiplier(real_multiplier)?;

        let activation = get_activation::<i32>(op_code);
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

    pub fn registration<'a, T: ArrayElem<T>>() -> BLiteRegistration<'a, T> {
        BLiteRegistration::new(Self::OPCODE, Self::eval::<T>, NotInitialize)
    }

    pub fn eval<'a, T: ArrayElem<T>>(
        _context: &BLiteContext,
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
        let input = tensors[idx_input]._b_tensor()?.borrow();

        let idx_filter = node.inputs[1] as usize;
        let filter = tensors[idx_filter]._b_tensor()?.borrow();

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output]._b_tensor()?.borrow_mut();

        let idx_bias = node.inputs[2];

        let batches = 1usize;
        let output_depth = filter.dims[filter.dims.len() - 2] as usize;
        let accum_depth = filter.dims[filter.dims.len() - 1] as usize;

        if idx_bias >= 0 {
            let bias = tensors[idx_bias as usize]._i32_tensor()?.borrow();

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

    #[inline(always)]
    fn kernel<'a, T: ArrayElem<T>>(
        input_data: &[T],
        filter_data: &[T],
        bias_data: Option<&[i32]>,
        output_data: &mut [T],
        // for quantization
        input_offset: i32,
        filter_offset: i32,
        output_offset: i32,
        output_depth: usize,
        output_multiplier: i32,
        output_shift: i32,
        //
        accum_depth: usize,
        batches: usize,
        activation: Option<fn(i32) -> i32>,
    ) -> Result<()> {
        for batch in 0usize..batches {
            for out_d in 0usize..output_depth {
                let mut total = 0;
                for acc_d in 0usize..accum_depth {
                    let input_val =
                        AsPrimitive::<i8>::as_(input_data[batch * accum_depth + acc_d]) as i32;
                    let filter_val =
                        AsPrimitive::<i8>::as_(filter_data[out_d * accum_depth + acc_d]) as i32;
                    total += (input_val + input_offset) * (filter_val + filter_offset);
                }

                if let Some(bias_data) = bias_data {
                    total += bias_data[out_d];
                }

                total = multiply_by_quantized_multiplier(total, output_multiplier, output_shift)?;
                // TODO: this code should be placed in the above loop.
                if let Some(activation) = activation {
                    total = activation(total);
                }

                total += output_offset;
                total = max(total, core::i8::MIN as i32);
                total = min(total, core::i8::MAX as i32);

                output_data[batch * output_depth + out_d] =
                    FromPrimitive::from_i8(total as i8).unwrap();
            }
        }

        Ok(())
    }
}
