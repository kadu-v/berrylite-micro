use crate::micro_context::BLiteContext;
use crate::micro_erros::BLiteStatus;
use crate::micro_graph::BLiteNode;

pub trait Regstration<'a, T> {
    fn eval(
        self,
        context: BLiteContext,
        node: &'a BLiteNode<'a>,
    ) -> BLiteStatus;
}

#[derive(Debug)]
pub enum Ops {
    Dummy,
}

impl<'a, R> Regstration<'a, R> for Ops {
    fn eval(
        self,
        context: BLiteContext,
        node: &'a BLiteNode<'a>,
    ) -> BLiteStatus {
        todo!()
    }
}
