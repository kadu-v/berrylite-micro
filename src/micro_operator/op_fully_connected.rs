use crate::micro_context::BLiteContext;
use crate::micro_erros::Result;
use crate::micro_graph::BLiteNode;
use crate::micro_registration::BLiteRegstration;

#[derive(Debug, Clone, Copy)]
pub struct OpFullyConnected {
    op_code: i32,
}

impl OpFullyConnected {
    pub fn regstration() -> BLiteRegstration {
        BLiteRegstration::new(9, Self::eval)
    }

    pub fn eval<'a>(
        context: &BLiteContext,
        node: &BLiteNode<'a>,
    ) -> Result<()> {
        todo!()
    }
}
