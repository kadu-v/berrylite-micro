use crate::micro_erros::{BLiteError::*, Result};
use crate::micro_operator::BLiteOperator;
use crate::micro_registration::BLiteRegstration;
use crate::tflite_schema_generated::tflite::BuiltinOperator;

#[derive(Debug, Clone, Copy)]
pub struct BLiteOpResorlver<const N: usize> {
    idx: usize,
    operators: [Option<BLiteOperator>; N],
}

impl<const N: usize> BLiteOpResorlver<N> {
    pub const fn new() -> Self {
        Self {
            idx: 0,
            operators: [None; N],
        }
    }

    pub fn find_op(
        &self,
        op: &BuiltinOperator,
    ) -> Result<BLiteOperator> {
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
        operator: BLiteOperator,
    ) -> Result<()> {
        if self.idx >= self.operators.len() {
            return Err(OpIndexOutOfBound);
        }

        self.operators[self.idx] = Some(operator);
        return Ok(());
    }
}
