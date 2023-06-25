pub type Result<T> = core::result::Result<T, BLiteError>;

#[derive(Debug)]
pub enum BLiteError {
    // allocator errors
    AllocationFailed,

    // micro arrray errors
    NotMatchSize,

    // micro graph errors
    CreateGraphFailed,
    NotFoundTensor,
    NotFoundBufferData,
}
