use num_traits::{AsPrimitive, FromPrimitive};

use crate::kernel::micro_builtin_options::{BLiteBuiltinOption, BLiteBuiltinOption::*};
use crate::kernel::micro_operator::BLiteOperator;
use crate::kernel::utils::quantization::{dequantize, quantize};
use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::{ArrayElem, BLiteQuantizationParams};
use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteError::{self, NotFoundOption};
use crate::micro_erros::Result;
use crate::micro_node::BLiteNode;
use crate::micro_registration::BLiteRegistration;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub struct OpSoftMaxInt8 {}

impl OpSoftMaxInt8 {
    const OPCODE: i32 = 25;

    pub fn softmax_int8<'a, T: ArrayElem<T>, S: ArenaAllocator>() -> BLiteOperator<'a, T, S> {
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
        let builtin_option = op.builtin_options_as_softmax_options();
        let mut beta = 1.0;
        if let Some(builtin_option) = builtin_option {
            beta = builtin_option.beta();
        }

        let input_idx = op.inputs().unwrap().get(0) as usize;
        let (input_scale, input_zero_point) = {
            let Some(BLiteQuantizationParams { scale, zero_point }) =
                tensors[input_idx]._t()?.borrow().quant_params
            else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale[0], zero_point[0] as i32)
        };

        let output_idx = op.outputs().unwrap().get(0) as usize;
        let (output_scale, output_zero_point) = {
            let Some(BLiteQuantizationParams { scale, zero_point }) =
                tensors[output_idx]._t()?.borrow().quant_params
            else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale[0], zero_point[0] as i32)
        };

        Ok(BLiteBuiltinOption::QuantizedSoftMaxOptions {
            beta,
            input_scale,
            input_zero_point,
            output_scale,
            output_zero_point,
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
        let input = tensors[idx_input]._t()?.borrow();

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output]._t()?.borrow_mut();

        let trailing_dims = input.dims.len() - 1;
        let outer_size = input.dims[0..trailing_dims]
            .iter()
            .fold(1, |x, &acc| x * acc);
        let depth = input.dims[input.dims.len() - 1];
        let QuantizedSoftMaxOptions {
            beta,
            input_scale,
            input_zero_point,
            output_scale,
            output_zero_point,
        } = builtin_option
        else {
            return Err(NotFoundOption);
        };

        for i in 0..outer_size {
            let mut max = core::f32::MIN;
            for c in 0..depth {
                let input_v = AsPrimitive::<i32>::as_(input.data[(i * depth + c) as usize]);
                let dequantize_v = dequantize(input_scale, input_zero_point, input_v)?;
                if dequantize_v > max {
                    max = dequantize_v;
                }
            }

            let mut sum: f32 = 0.0;
            for c in 0..depth {
                let idx = (i * depth + c) as usize;
                let input_v = AsPrimitive::<i32>::as_(input.data[idx as usize]);
                let dequantize_v = dequantize(input_scale, input_zero_point, input_v)?;
                let exp_c = ((dequantize_v - max) * beta).exp();
                let mut quantize_exp_c = quantize(output_scale, output_zero_point, exp_c)?;

                quantize_exp_c = core::cmp::max(quantize_exp_c, T::MIN.as_());
                quantize_exp_c = core::cmp::min(quantize_exp_c, T::MAX.as_());
                output.data[idx] = FromPrimitive::from_i32(quantize_exp_c).unwrap();
                sum = sum + exp_c;
            }

            for c in 0..depth {
                let idx = (i * depth + c) as usize;
                let out_v = AsPrimitive::<i32>::as_(output.data[idx]);
                let dequantize_out_v = dequantize(output_scale, output_zero_point, out_v)?;
                let v = dequantize_out_v / sum;
                let quantize_out_v = quantize(output_scale, output_zero_point, v)?;
                output.data[idx] = FromPrimitive::from_i32(quantize_out_v).unwrap();
            }
        }

        Ok(())
    }
}
