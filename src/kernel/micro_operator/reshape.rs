use crate::kernel::micro_builtin_options::{
    BLiteBuiltinOption, BLiteBuiltinOption::*,
};
use crate::micro_array::ArrayElem;
use crate::micro_context::BLiteContext;
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
        todo!()
    }
}
