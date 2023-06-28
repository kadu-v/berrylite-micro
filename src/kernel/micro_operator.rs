pub mod op_fully_connected;

use crate::micro_registration::BLiteRegstration;
use core::fmt::Debug;
use op_fully_connected::OpFullyConnected;

#[derive(Debug, Clone, Copy)]
pub struct BLiteOperator<T>
where
    T: Debug + Clone + Copy,
{
    regstration: BLiteRegstration<T>,
}

impl<T> BLiteOperator<T>
where
    T: Debug + Clone + Copy,
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
