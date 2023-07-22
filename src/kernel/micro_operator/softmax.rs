use crate::kernel::micro_builtin_options::{
    BLiteBuiltinOption, BLiteBuiltinOption::*,
};
use crate::micro_array::ArrayElem;
use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteError::NotFoundOption;
use crate::micro_erros::Result;
use crate::micro_node::BLiteNode;
use crate::micro_registration::BLiteRegistration;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::fmt::Debug;
// Importing Float module is necessary to calculate the exponential of a f32 value.
#[allow(unused_imports)]
use num_traits::Float;

use crate::kernel::micro_operator::BLiteOperator;

#[derive(Debug, Clone, Copy)]
pub struct SoftMax {}

impl SoftMax {
    const OPCODE: i32 = 25;

    pub fn softmax<T: ArrayElem<T>>() -> BLiteOperator<T> {
        BLiteOperator {
            registration: Self::registration(),
            parser: Self::parser,
        }
    }

    pub fn parser<T: ArrayElem<T>>(
        op: Operator,
        _tensors: &mut [BLiteTensor<'_, T>],
    ) -> Result<BLiteBuiltinOption<T>> {
        let builtin_option =
            op.builtin_options_as_softmax_options();
        let mut beta = 1.0;
        if let Some(builtin_option) = builtin_option {
            beta = builtin_option.beta();
        }
        Ok(BLiteBuiltinOption::SoftMaxOptions { beta })
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

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output].borrow_mut();

        let trailing_dims =
            &input.dims[0..input.dims.len() - 1];
        let outer_size =
            trailing_dims.iter().fold(1, |x, &acc| x * acc);
        let depth = input.dims[input.dims.len() - 1];
        let SoftMaxOptions { beta } = builtin_option else {
            return Err(NotFoundOption);
        };

        for i in 0..outer_size {
            let mut max = Default::default();
            for c in 0..depth {
                let input_v =
                    input.data[(i * depth + c) as usize];
                if input_v > max {
                    max = input_v;
                }
            }

            let mut sum: T = Default::default();
            for c in 0..depth {
                let idx = (i * depth + c) as usize;
                let exp_c =
                    ((input.data[idx] - max).into() * beta)
                        .exp()
                        .into();
                output.data[idx] = exp_c;
                sum = sum + exp_c;
            }

            for c in 0..depth {
                let idx = (i * depth + c) as usize;
                output.data[idx] = output.data[idx] / sum;
            }
        }
        Ok(())
    }
}
