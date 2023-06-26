use core::fmt::Debug;
use core::task::Context;

use crate::micro_context::BLiteContext;
use crate::micro_erros::Result;
use crate::micro_graph::BLiteNode;
pub struct Regstration {
    eval: for<'a> fn(
        context: &BLiteContext,
        node: &BLiteNode<'a>,
    ) -> Result<()>,
}

impl Regstration {
    pub fn new(
        eval: for<'a> fn(
            context: &BLiteContext,
            node: &BLiteNode<'a>,
        ) -> Result<()>,
    ) -> Self {
        Self { eval }
    }

    pub fn default() -> Self {
        Self { eval: Self::eval }
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

impl Debug for Regstration {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "Op {{ eval..., }}")?;
        Ok(())
    }
}
