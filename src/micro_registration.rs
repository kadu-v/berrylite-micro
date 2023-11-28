use core::fmt::Debug;

use crate::kernel::micro_builtin_options::BLiteBuiltinOption;
use crate::micro_array::ArrayElem;
use crate::micro_context::BLiteContext;
use crate::micro_errors::Result;
use crate::micro_node::BLiteNode;
use crate::micro_tensor::BLiteTensor;

#[derive(Clone, Copy)]
pub struct BLiteRegistration<'a, T>
where
    T: ArrayElem<T>,
{
    pub op_code: i32,
    pub eval: fn(
        context: &BLiteContext,
        tensors: &'a mut [BLiteTensor<'a, T>],
        node: &BLiteNode<'a>,
        builtin_option: BLiteBuiltinOption<T>,
    ) -> Result<()>,
    pub builtin_option: BLiteBuiltinOption<'a, T>,
}

impl<'a, T: ArrayElem<T>> BLiteRegistration<'a, T> {
    pub fn new(
        op_code: i32,
        eval: fn(
            _context: &BLiteContext,
            tensors: &'a mut [BLiteTensor<'a, T>],
            node: &BLiteNode<'a>,
            builtin_option: BLiteBuiltinOption<T>,
        ) -> Result<()>,
        builtin_option: BLiteBuiltinOption<'a, T>,
    ) -> Self {
        Self {
            op_code,
            eval,
            builtin_option,
        }
    }

    pub fn call_eval(
        &self,
        tensors: &'a mut [BLiteTensor<'a, T>],
        context: &BLiteContext,
        node: &BLiteNode<'a>,
        builtin_option: BLiteBuiltinOption<T>,
    ) -> Result<()> {
        let eval = self.eval;
        eval(context, tensors, node, builtin_option)
    }
}

impl<'a, T: ArrayElem<T>> Debug for BLiteRegistration<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Registration {{ op_code: {}, eval: {:p}, builtin_option: {:?} }}",
            self.op_code, self.eval, self.builtin_option
        )?;
        Ok(())
    }
}
