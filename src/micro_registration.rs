use core::fmt::Debug;

use crate::micro_context::BLiteContext;
use crate::micro_erros::Result;
use crate::micro_graph::BLiteNode;

#[derive(Clone, Copy)]
pub struct BLiteRegstration {
    pub op_code: i32,
    pub eval: for<'a> fn(
        context: &BLiteContext,
        node: &BLiteNode<'a>,
    ) -> Result<()>,
}

impl BLiteRegstration {
    pub fn new(
        op_code: i32,
        eval: for<'a> fn(
            context: &BLiteContext,
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
        context: &BLiteContext,
        node: &BLiteNode<'a>,
    ) -> Result<()> {
        Ok(())
    }

    pub fn call_eval<'a>(
        &self,
        context: &BLiteContext,
        node: &BLiteNode<'a>,
    ) -> Result<()> {
        let eval = self.eval;
        eval(context, node)
    }
}

impl Debug for BLiteRegstration {
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
