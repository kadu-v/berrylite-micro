use core::fmt::Debug;

use crate::micro_context::BLiteContext;
use crate::micro_erros::Result;
use crate::micro_graph::BLiteNode;
use crate::micro_registration::BLiteRegstration;

#[derive(Debug, Clone, Copy)]
pub struct OpFullyConnected {
    op_code: i32,
}

impl OpFullyConnected {
    pub fn regstration<T: Debug + Clone + Copy>(
    ) -> BLiteRegstration<T> {
        BLiteRegstration::new(9, Self::eval::<T>)
    }

    pub fn eval<'a, T: Debug>(
        context: &BLiteContext<'a, T>,
        node: &BLiteNode<'a>,
    ) -> Result<()> {
        // let tensors = context.get_tensors();
        // let idx_input = node.inputs[0] as usize;
        // let input = &tensors[idx_input];
        // let data = &input.data;
        // let data = *data;

        // let idx_filter = node.inputs[1] as usize;
        // let idx_bias = node.inputs[2] as usize;
        // let idx_output = node.outputs[0] as usize;

        todo!();

        Ok(())
    }
}
