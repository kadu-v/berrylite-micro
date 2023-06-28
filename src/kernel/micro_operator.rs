pub mod op_fully_connected;

use crate::micro_array::ArrayElem;
use crate::micro_registration::BLiteRegstration;
use core::fmt::Debug;
use core::ops::Add;
use op_fully_connected::OpFullyConnected;

#[derive(Debug, Clone, Copy)]
pub struct BLiteOperator<T>
where
    T: ArrayElem,
{
    regstration: BLiteRegstration<T>,
}

impl<T> BLiteOperator<T>
where
    T: Debug + Clone + Copy + Add,
{
    pub fn get_op_code(&self) -> i32 {
        self.regstration.op_code
    }

    pub fn get_regstration(&self) -> BLiteRegstration<T> {
        self.regstration
    }

    pub fn fully_connected() -> Self {
        Self {
            regstration: OpFullyConnected::regstration(),
        }
    }
}
