use core::fmt::Debug;

use crate::kernel::micro_operator::BLiteOperator;
use crate::micro_erros::{BLiteError::*, Result};
use crate::tflite_schema_generated::tflite::BuiltinOperator;

#[derive(Debug, Clone)]
pub struct BLiteOpResorlver<const N: usize, T>
where
    T: Debug + Clone + Copy,
{
    idx: usize,
    operators: [Option<BLiteOperator<T>>; N],
}

impl<const N: usize, T: Debug + Clone + Copy>
    BLiteOpResorlver<N, T>
{
    pub const fn new() -> Self {
        Self {
            idx: 0,
            operators: [None; N],
        }
    }

    pub fn find_op(
        &self,
        op: &BuiltinOperator,
    ) -> Result<BLiteOperator<T>> {
        for operator in self.operators {
            if let Some(blite_op) = operator {
                if blite_op.get_op_code() == op.0 {
                    return Ok(blite_op);
                }
            }
        }
        Err(NotFoundOperator)
    }

    pub fn add_op(
        &mut self,
        operator: BLiteOperator<T>,
    ) -> Result<()> {
        if self.idx >= self.operators.len() {
            return Err(OpIndexOutOfBound);
        }

        self.operators[self.idx] = Some(operator);
        return Ok(());
    }
}
