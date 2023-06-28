use crate::micro_array::ArrayElem;
use crate::micro_context::BLiteContext;
use crate::micro_erros::Result;
use crate::micro_graph::BLiteNode;
use crate::micro_registration::BLiteRegstration;
use crate::micro_tensor::BLiteTensor;
use core::fmt::Debug;
use core::ops::Add;

#[derive(Debug, Clone, Copy)]
pub struct OpFullyConnected {
    op_code: i32,
}

impl OpFullyConnected {
    pub fn regstration<T: ArrayElem>() -> BLiteRegstration<T>
    {
        BLiteRegstration::new(9, Self::eval::<T>)
    }

    pub fn eval<'a, T: ArrayElem>(
        context: &BLiteContext<'a, T>,
        tensors: &'a mut [BLiteTensor<'a, T>],
        node: &BLiteNode<'a>,
    ) -> Result<()> {
        let idx_input = node.inputs[0] as usize;
        let input = tensors[idx_input].borrow();

        let idx_filter = node.inputs[1] as usize;
        let filter = tensors[idx_filter].borrow();

        // let idx_bias = node.inputs[2] as usize;
        let idx_output = node.outputs[0] as usize;
        let mut output = tensors[idx_output].borrow_mut();

        for i in 0..output.len() {
            output.data[i] = filter.data[0];
        }

        Ok(())
    }
}
