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
        writeln!(f, "Op {{ eval..., }}")?;
        Ok(())
    }
}
