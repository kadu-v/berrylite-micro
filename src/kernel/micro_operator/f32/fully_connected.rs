use crate::kernel::micro_activation::get_activation;
use crate::kernel::micro_builtin_options::{BLiteBuiltinOption, BLiteBuiltinOption::*};
use crate::kernel::utils::types::flat_skip_dims;
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
pub struct OpFullyConnected {}

impl OpFullyConnected {
    const OPCODE: i32 = 9;

    pub fn fully_connected<'a, T: ArrayElem<T>, S: ArenaAllocator>() -> BLiteOperator<'a, T, S> {
        BLiteOperator {
            registration: Self::registration(),
            parser: Self::parser,
        }
    }

    pub fn parser<'a, T: ArrayElem<T>>(
        _allocator: &mut impl ArenaAllocator,
        op: Operator,
        _tensors: &mut [BLiteTensor<'a, T>],
    ) -> Result<BLiteBuiltinOption<'a, T>> {
        let builtin_option = op.builtin_options_as_fully_connected_options();
        let mut op_code = -1;
        if let Some(builtin_option) = builtin_option {
            op_code = builtin_option.fused_activation_function().0 as i32;
        }
        let activation = get_activation::<T>(op_code);
        Ok(BLiteBuiltinOption::FullyConnectedOptions {
            op_code,
            activation,
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
        let idx_input = node.inputs[0];
        let input = tensors[idx_input as usize]._t()?.borrow();

        let idx_filter = node.inputs[1];
        let filter = tensors[idx_filter as usize]._t()?.borrow();

        let idx_output = node.outputs[0];
        let mut output = tensors[idx_output as usize]._t()?.borrow_mut();
        let output_data_size = output.data.len();

        let idx_bias = node.inputs[2];

        let activation = match builtin_option {
            FullyConnectedOptions {
                op_code: _,
                activation,
            } => activation,
            NotInitialize => return Err(NotInitializeActivation),
            _ => return Err(NotCompatibleOption),
        };

        // TODO:

        let batches = flat_skip_dims(output.dims, output.dims.len() - 1);
        let output_depth = filter.dims[filter.dims.len() - 2];
        let accum_depth = filter.dims[filter.dims.len() - 1];
        if idx_bias >= 0 {
            let bias = tensors[idx_bias as usize]._t()?.borrow();
            Self::kernel(
                input.data,
                Some(&bias.data),
                filter.data,
                output.data,
                batches,
                output_data_size,
                output_depth,
                accum_depth,
                activation,
            )
        } else {
            Self::kernel(
                input.data,
                None,
                filter.data,
                output.data,
                batches,
                output_data_size,
                output_depth,
                accum_depth,
                activation,
            )
        }
    }

    #[inline(always)]
    pub fn kernel<T: ArrayElem<T>>(
        input_data: &[T],
        bias_data: Option<&[T]>,
        filter_data: &[T],
        output_data: &mut [T],
        batches: i32,
        output_data_size: usize,
        output_depth: i32,
        accum_depth: i32,
        activation: Option<fn(T) -> T>,
    ) -> Result<()> {
        for batch in 0..batches as usize {
            for out_d in 0..output_depth as usize {
                let mut total: T = Default::default();
                for acc_d in 0..accum_depth as usize {
                    total += input_data[batch * accum_depth as usize + acc_d]
                        * filter_data[out_d * accum_depth as usize + acc_d];
                }

                output_data[batch * output_depth as usize + out_d] = total;
                if let Some(bias_data) = bias_data {
                    let bias = bias_data[out_d];
                    output_data[batch * output_depth as usize + out_d] += bias;
                }
            }
        }
        if let Some(activation) = activation {
            for i in 0..output_data_size {
                output_data[i] = activation(output_data[i]);
            }
        }

        Ok(())
    }
}
