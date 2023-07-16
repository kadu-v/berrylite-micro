use crate::builtin_op_data;
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
pub struct Conv2D {}

impl Conv2D {
    const OPCODE: i32 = 3;

    pub fn conv2d<T: ArrayElem<T>>() -> BLiteOperator<T> {
        BLiteOperator {
            registration: Self::registration(),
            parser: Self::parser,
        }
    }

    pub fn parser<T: ArrayElem<T>>(
        op: Operator,
    ) -> Result<BLiteBuiltinOption<T>> {
        let builtin_option =
            op.builtin_options_as_conv_2_doptions();
        let mut op_code = -1;
        let Some(builtin_option) = builtin_option else {
            return Err(NotFoundOption);
        };
        op_code = builtin_option
            .fused_activation_function()
            .0 as i32;
        let padding = builtin_option.padding().0 as usize;
        let activation = get_activation::<T>(op_code);
        let stride_w = builtin_option.stride_w();
        let stride_h = builtin_option.stride_h();
        let dilation_w_factor =
            builtin_option.dilation_w_factor();
        let dilation_h_factor =
            builtin_option.dilation_h_factor();
        Ok(BLiteBuiltinOption::Conv2DOptions {
            op_code,
            activation,
            padding,
            stride_w,
            stride_h,
            dilation_w_factor,
            dilation_h_factor,
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
        todo!("conv2d eval")
    }
}
