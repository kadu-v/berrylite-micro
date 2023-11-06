use core::fmt::Debug;

use crate::kernel::micro_operator::BLiteOperator;
use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::ArrayElem;
use crate::micro_errors::{BLiteError::*, Result};

#[derive(Debug, Clone)]
pub struct BLiteOpResolver<'a, const N: usize, T, S>
where
    T: ArrayElem<T>,
    S: ArenaAllocator,
{
    idx: usize,
    operators: [Option<BLiteOperator<'a, T, S>>; N],
}

impl<'a, const N: usize, T: ArrayElem<T>, S: ArenaAllocator> BLiteOpResolver<'a, N, T, S> {
    const BLITE_OP_DEFAULT: Option<BLiteOperator<'a, T, S>> = None;

    pub const fn new() -> Self {
        Self {
            idx: 0,
            operators: [Self::BLITE_OP_DEFAULT; N],
        }
    }

    pub fn find_op(&self, op_code: i32) -> Result<&'a BLiteOperator<T, S>> {
        for operator in &self.operators {
            if let Some(blite_op) = operator {
                let blite_op_code = blite_op.get_op_code();
                if blite_op_code == op_code {
                    return Ok(blite_op);
                }
            }
        }
        Err(NotFoundOperator(op_code))
    }

    pub fn add_op(&mut self, operator: BLiteOperator<'a, T, S>) -> Result<()> {
        if self.idx >= self.operators.len() {
            return Err(OpIndexOutOfBound);
        }

        self.operators[self.idx] = Some(operator);
        self.idx += 1;
        return Ok(());
    }
}
