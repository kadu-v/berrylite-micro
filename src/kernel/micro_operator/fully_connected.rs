use crate::kernel::micro_activation::get_activation;
use crate::kernel::micro_builtin_options::{
    BLiteBuiltinOption, BLiteBuiltinOption::*,
};
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

    pub fn fully_connected<T: ArrayElem<T>>(
    ) -> BLiteOperator<T> {
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
            op.builtin_options_as_fully_connected_options();
        let mut op_code = -1;
        if let Some(builtin_option) = builtin_option {
            op_code = builtin_option
                .fused_activation_function()
                .0 as i32;
        }
        let activation = get_activation::<T>(op_code);
        Ok(BLiteBuiltinOption::FullyConnectedOptions {
            op_code,
            activation,
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

        let idx_filter = node.inputs[1] as usize;
        let filter = tensors[idx_filter].borrow();

        let idx_bias = node.inputs[2] as usize;
        let bias = tensors[idx_bias].borrow();

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output].borrow_mut();

        let activation = match builtin_option {
            FullyConnectedOptions {
                op_code: _,
                activation,
            } => activation,
            NotInitialize => {
                return Err(NotInitializeActivation)
            }
            _ => return Err(NotCompatibleOption),
        };

        // TODO:
        let batches = 1usize;
        let output_depth =
            filter.dims[filter.dims.len() - 2] as usize;
        let accum_depth =
            filter.dims[filter.dims.len() - 1] as usize;

        for batch in 0..batches {
            for out_d in 0..output_depth {
                let mut total: T = Default::default();
                for acc_d in 0..accum_depth {
                    total = total
                        + input.data
                            [batch * accum_depth + acc_d]
                            * filter.data[out_d
                                * accum_depth
                                + acc_d];
                }
                output.data[batch * output_depth + out_d] =
                    total + bias.data[out_d];
            }
        }
        if let Some(activation) = activation {
            for i in 0..output.data.len() {
                output.data[i] = activation(output.data[i]);
            }
        }

        Ok(())
    }
}
