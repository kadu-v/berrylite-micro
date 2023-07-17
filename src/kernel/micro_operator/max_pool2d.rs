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
use crate::tflite_schema_generated::tflite::{
    BuiltinOptions, Operator,
};
use core::fmt::Debug;

use crate::kernel::micro_operator::BLiteOperator;

#[derive(Debug, Clone, Copy)]
pub struct MaxPool2D {}

impl MaxPool2D {
    const OPCODE: i32 = 17;

    pub fn max_pool2d<T: ArrayElem<T>>() -> BLiteOperator<T>
    {
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
            op.builtin_options_as_pool_2_doptions();
        let mut op_code = -1;
        let Some(builtin_option) = builtin_option else {
            return Err(NotFoundOption);
        };
        op_code = builtin_option
            .fused_activation_function()
            .0 as i32;
        let activation = get_activation::<T>(op_code);
        let padding = builtin_option.padding().0 as usize;
        let stride_w = builtin_option.stride_w();
        let stride_h = builtin_option.stride_h();
        let filter_width = builtin_option.filter_height();
        let filter_height = builtin_option.filter_height();
        Ok(BLiteBuiltinOption::MaxPool2DOptions {
            op_code,
            activation,
            padding,
            stride_w,
            stride_h,
            filter_width,
            filter_height,
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
        todo!()
    }
}
