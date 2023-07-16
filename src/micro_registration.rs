use core::fmt::Debug;

use crate::kernel::micro_builtin_options::BLiteBuiltinOption;
use crate::micro_array::ArrayElem;
use crate::micro_context::BLiteContext;
use crate::micro_erros::Result;
use crate::micro_node::BLiteNode;
use crate::micro_tensor::BLiteTensor;

#[derive(Clone, Copy)]
pub struct BLiteRegistration<T>
where
    T: ArrayElem<T>,
{
    pub op_code: i32,
    pub eval: for<'a> fn(
        context: &BLiteContext<'a, T>,
        tensors: &'a mut [BLiteTensor<'a, T>],
        node: &BLiteNode<'a>,
        builtin_option: BLiteBuiltinOption<T>,
    ) -> Result<()>,
    pub builtin_option: BLiteBuiltinOption<T>,
}

impl<T: ArrayElem<T>> BLiteRegistration<T> {
    pub fn new(
        op_code: i32,
        eval: for<'a> fn(
            context: &BLiteContext<'a, T>,
            tensors: &'a mut [BLiteTensor<'a, T>],
            node: &BLiteNode<'a>,
            builtin_option: BLiteBuiltinOption<T>,
        ) -> Result<()>,
        builtin_option: BLiteBuiltinOption<T>,
    ) -> Self {
        Self {
            op_code,
            eval,
            builtin_option,
        }
    }

    pub fn call_eval<'a>(
        &self,
        tensors: &'a mut [BLiteTensor<'a, T>],
        context: &BLiteContext<'a, T>,
        node: &BLiteNode<'a>,
        builtin_option: BLiteBuiltinOption<T>,
    ) -> Result<()> {
        let eval = self.eval;
        eval(context, tensors, node, builtin_option)
    }
}

impl<T: ArrayElem<T>> Debug for BLiteRegistration<T> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "Registration {{ op_code: {}, eval:..., builtin_option: {:?} }}",
            self.op_code,
            self.builtin_option
        )?;
        Ok(())
    }
}
