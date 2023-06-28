pub mod op_fully_connected;

use crate::micro_array::ArrayElem;
use crate::micro_erros::Result;
use crate::micro_registration::BLiteRegstration;
use crate::tflite_schema_generated::tflite::Operator;
use core::fmt::Debug;

use super::micro_builtin_options::BLiteBuiltinOption;

#[derive(Clone, Copy)]
pub struct BLiteOperator<T>
where
    T: ArrayElem<T>,
{
    regstration: BLiteRegstration<T>,
    parser:
        fn(op: Operator) -> Result<BLiteBuiltinOption<T>>,
}

impl<T> BLiteOperator<T>
where
    T: ArrayElem<T>,
{
    pub fn get_op_code(&self) -> i32 {
        self.regstration.op_code
    }

    pub fn get_regstration(&self) -> BLiteRegstration<T> {
        self.regstration
    }

    // pub fn get_parser(
    //     &self,
    // ) -> fn(op: Operator) -> BLiteBuiltinOption<T> {
    //     self.parser
    // }
}

impl<T: ArrayElem<T>> Debug for BLiteOperator<T> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(
            f,
            "Operator {{ registration: {:?}, parse:...}}",
            self.regstration,
        )?;
        Ok(())
    }
}
