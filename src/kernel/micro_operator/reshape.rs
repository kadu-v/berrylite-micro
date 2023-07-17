use crate::kernel::micro_builtin_options::{
    BLiteBuiltinOption, BLiteBuiltinOption::*,
};
use crate::micro_array::ArrayElem;
use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteError::InCompatibelShape;
use crate::micro_erros::Result;
use crate::micro_node::BLiteNode;
use crate::micro_registration::BLiteRegistration;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::fmt::Debug;

use crate::kernel::micro_operator::BLiteOperator;

#[derive(Debug, Clone, Copy)]
pub struct Reshape {}

impl Reshape {
    const OPCODE: i32 = 22;

    pub fn reshape<T: ArrayElem<T>>() -> BLiteOperator<T> {
        BLiteOperator {
            registration: Self::registration(),
            parser: Self::parser,
        }
    }

    pub fn parser<T: ArrayElem<T>>(
        op: Operator,
        tensors: &mut [BLiteTensor<'_, T>],
    ) -> Result<BLiteBuiltinOption<T>> {
        Ok(BLiteBuiltinOption::ReshapeOptions {})
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

        // shape checking
        let input_elems =
            input.dims.iter().fold(1, |x, acc| x * acc);
        let output_elems =
            output.dims.iter().fold(1, |x, acc| x * acc);
        if input_elems != output_elems {
            return Err(InCompatibelShape(
                input_elems,
                output_elems,
            ));
        }

        for i in 0..input_elems {
            output.data[i as usize] =
                input.data[i as usize];
        }
        Ok(())
    }
}
