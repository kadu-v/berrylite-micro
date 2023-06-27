pub mod op_fully_connected;

use crate::micro_registration::BLiteRegstration;
use op_fully_connected::OpFullyConnected;

#[derive(Debug, Clone, Copy)]
pub struct BLiteOperator {
    regstration: BLiteRegstration,
}

impl BLiteOperator {
    pub fn get_op_code(&self) -> i32 {
        self.regstration.op_code
    }

    pub fn get_regstration(&self) -> BLiteRegstration {
        self.regstration
    }

    pub fn fully_connected() -> Self {
        Self {
            regstration: OpFullyConnected::regstration(),
        }
    }
}
