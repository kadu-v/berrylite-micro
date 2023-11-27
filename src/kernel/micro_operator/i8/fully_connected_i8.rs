use num_traits::{AsPrimitive, FromPrimitive};

use crate::kernel::micro_activation::calculate_fused_activation_range_quantized;
use crate::kernel::micro_builtin_options::{BLiteBuiltinOption, BLiteBuiltinOption::*};
use crate::kernel::utils::quantization::{
    get_quantized_convolution_multiplier, multiply_by_quantized_multiplier, quantize_multiplier,
};
use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::{ArrayElem, BLiteQuantizationParams};
use crate::micro_context::BLiteContext;
use crate::micro_errors::BLiteError::{self, *};
use crate::micro_errors::Result;
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
        _allocator: &mut impl ArenaAllocator,
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
            let Some(BLiteQuantizationParams { scale, zero_point }) =
                tensors[input_idx]._t()?.borrow().quant_params
            else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale[0], zero_point[0] as i32)
        };

        let filter_idx = op.inputs().unwrap().get(1) as usize;
        let (filter_scale, filter_zero_point) = {
            let Some(BLiteQuantizationParams { scale, zero_point }) =
                tensors[filter_idx]._t()?.borrow().quant_params
            else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            (scale[0], zero_point[0] as i32)
        };

        let bias_idx = op.inputs().unwrap().get(2);
        let bias_scale = if bias_idx >= 0 {
            let Some(BLiteQuantizationParams { scale, .. }) =
                tensors[bias_idx as usize]._i32()?.borrow().quant_params
            else {
                return Err(BLiteError::NotFoundQuantParams);
            };
            Some(scale[0])
        } else {
            None
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
        let (fused_activation_min, fused_activation_max) =
            calculate_fused_activation_range_quantized::<T>(
                output_scale,
                output_zero_point,
                op_code,
            )?;

        //This computations is corresponded to CalculateOpDataFullyConnected
        let real_multiplier = get_quantized_convolution_multiplier(
            input_scale,
            filter_scale,
            output_scale,
            bias_scale,
        )?;
        let (output_multiplier, output_shift) = quantize_multiplier(real_multiplier)?;

        Ok(BLiteBuiltinOption::QuantizedFullyConnectedOptions {
            op_code,
            fused_activation_min,
            fused_activation_max,
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
        dbg!("x");

        let QuantizedFullyConnectedOptions {
            op_code: _,
            fused_activation_min,
            fused_activation_max,
            input_offset,
            filter_offset,
            output_offset,
            output_multiplier,
            output_shift,
        } = builtin_option
        else {
            return Err(NotInitializeActivation);
        };

        let idx_input = node.inputs[0] as usize;
        let input = tensors[idx_input]._t()?.borrow();

        let idx_filter = node.inputs[1] as usize;
        let filter = tensors[idx_filter]._t()?.borrow();

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output]._t()?.borrow_mut();

        let idx_bias = node.inputs[2];

        let batches = 1usize;
        let output_depth = filter.dims[filter.dims.len() - 2] as usize;
        let accum_depth = filter.dims[filter.dims.len() - 1] as usize;

        if idx_bias >= 0 {
            let bias = tensors[idx_bias as usize]._i32()?.borrow();

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
                fused_activation_min,
                fused_activation_max,
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
                fused_activation_min,
                fused_activation_max,
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
        fused_activation_min: i32,
        fused_activation_max: i32,
    ) -> Result<()> {
        for batch in 0usize..batches {
            for out_d in 0usize..output_depth {
                let mut total = 0;
                for acc_d in 0usize..accum_depth {
                    let input_val =
                        AsPrimitive::<i32>::as_(input_data[batch * accum_depth + acc_d]);
                    let filter_val =
                        AsPrimitive::<i32>::as_(filter_data[out_d * accum_depth + acc_d]);
                    total += (input_val + input_offset) * (filter_val + filter_offset);
                }

                if let Some(bias_data) = bias_data {
                    total += bias_data[out_d];
                }

                total = multiply_by_quantized_multiplier(total, output_multiplier, output_shift)?;

                total += output_offset;
                total = max(total, fused_activation_min);
                total = min(total, fused_activation_max);

                output_data[batch * output_depth + out_d] = FromPrimitive::from_i32(total).unwrap();
            }
        }
        Ok(())
    }
}
