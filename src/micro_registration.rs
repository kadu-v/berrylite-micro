use core::fmt::Debug;

use crate::micro_array::ArrayElem;
use crate::micro_context::BLiteContext;
use crate::micro_erros::Result;
use crate::micro_graph::BLiteNode;
use crate::micro_tensor::BLiteTensor;

#[derive(Clone, Copy)]
pub struct BLiteRegstration<T>
where
    T: ArrayElem,
{
    pub op_code: i32,
    pub eval: for<'a> fn(
        context: &BLiteContext<'a, T>,
        tensors: &'a mut [BLiteTensor<'a, T>],
        node: &BLiteNode<'a>,
    ) -> Result<()>,
}

impl<T: ArrayElem> BLiteRegstration<T> {
    pub fn new(
        op_code: i32,
        eval: for<'a> fn(
            context: &BLiteContext<'a, T>,
            tensors: &'a mut [BLiteTensor<'a, T>],
            node: &BLiteNode<'a>,
        ) -> Result<()>,
    ) -> Self {
        Self { op_code, eval }
    }

    pub fn default() -> Self {
        Self {
            op_code: 0,
            eval: Self::eval,
        }
    }

    pub fn eval<'a>(
        context: &BLiteContext<'a, T>,
        tensors: &'a mut [BLiteTensor<'a, T>],
        node: &BLiteNode<'a>,
    ) -> Result<()> {
        Ok(())
    }

    pub fn call_eval<'a>(
        &self,
        tensors: &'a mut [BLiteTensor<'a, T>],
        context: &BLiteContext<'a, T>,
        node: &BLiteNode<'a>,
    ) -> Result<()> {
        let eval = self.eval;
        eval(context, tensors, node)
    }
}

impl<T: ArrayElem> Debug for BLiteRegstration<T> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "Op {{ op_code: {} eval..., }}",
            self.op_code
        )?;
        Ok(())
    }
}
