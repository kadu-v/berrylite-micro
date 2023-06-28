use core::fmt::Debug;

use crate::micro_context::BLiteContext;
use crate::micro_erros::Result;
use crate::micro_graph::BLiteNode;

#[derive(Clone, Copy)]
pub struct BLiteRegstration<T>
where
    T: Debug + Clone + Copy,
{
    pub op_code: i32,
    pub eval: for<'a> fn(
        context: &BLiteContext<'a, T>,
        node: &BLiteNode<'a>,
    ) -> Result<()>,
}

impl<T: Debug + Clone + Copy> BLiteRegstration<T> {
    pub fn new(
        op_code: i32,
        eval: for<'a> fn(
            context: &BLiteContext<'a, T>,
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
        node: &BLiteNode<'a>,
    ) -> Result<()> {
        Ok(())
    }

    pub fn call_eval<'a>(
        &self,
        context: &BLiteContext<'a, T>,
        node: &BLiteNode<'a>,
    ) -> Result<()> {
        let eval = self.eval;
        eval(context, node)
    }
}

impl<T: Debug + Clone + Copy> Debug
    for BLiteRegstration<T>
{
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
