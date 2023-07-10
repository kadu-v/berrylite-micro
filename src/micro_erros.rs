pub type Result<T> = core::result::Result<T, BLiteError>;

#[derive(Debug)]
pub enum BLiteError {
    // allocator errors
    FailedToAllocateMemory,

    // micro arrray errors
    NotMatchSize,

    // micro graph errors
    FailedToCreateGraph,
    NotFoundTensor,
    NotFoundBufferData,
    MissingRegistration,
    NotFoundRegistration,

    // micro operator resolver
    NotFoundOperator,
    OpIndexOutOfBound,

    // micro fully connected
    NotInitializeActivation,
}
