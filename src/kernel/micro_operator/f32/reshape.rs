use crate::kernel::micro_builtin_options::{BLiteBuiltinOption, BLiteBuiltinOption::*};
use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::ArrayElem;
use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteError::InCompatibleShape;
use crate::micro_erros::Result;
use crate::micro_node::BLiteNode;
use crate::micro_registration::BLiteRegistration;
use crate::micro_tensor::BLiteTensor;
use crate::tflite_schema_generated::tflite::Operator;
use core::fmt::Debug;

use crate::kernel::micro_operator::BLiteOperator;

#[derive(Debug, Clone, Copy)]
pub struct OpReshape {}

impl OpReshape {
    const OPCODE: i32 = 22;

    pub fn reshape<'a, T: ArrayElem<T>, S: ArenaAllocator>() -> BLiteOperator<'a, T, S> {
        BLiteOperator {
            registration: Self::registration(),
            parser: Self::parser,
        }
    }

    pub fn parser<'a, T: ArrayElem<T>>(
        _allocator: &mut impl ArenaAllocator,
        _op: Operator,
        _tensors: &mut [BLiteTensor<'a, T>],
    ) -> Result<BLiteBuiltinOption<'a, T>> {
        Ok(BLiteBuiltinOption::ReshapeOptions {})
    }

    pub fn registration<'a, T: ArrayElem<T>>() -> BLiteRegistration<'a, T> {
        BLiteRegistration::new(Self::OPCODE, Self::eval::<T>, NotInitialize)
    }

    pub fn eval<'a, T: ArrayElem<T>>(
        _context: &BLiteContext,
        tensors: &'a mut [BLiteTensor<'a, T>],
        node: &BLiteNode<'a>,
        _builtin_option: BLiteBuiltinOption<T>,
    ) -> Result<()> {
        let idx_input = node.inputs[0] as usize;
        let input = tensors[idx_input]._t()?.borrow();

        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output]._t()?.borrow_mut();

        // shape checking
        let input_elems = input.dims.iter().fold(1, |x, acc| x * acc);
        let output_elems = output.dims.iter().fold(1, |x, acc| x * acc);
        if input_elems != output_elems {
            return Err(InCompatibleShape(input_elems, output_elems));
        }

        for i in 0..input_elems {
            output.data[i as usize] = input.data[i as usize];
        }
        Ok(())
    }
}
