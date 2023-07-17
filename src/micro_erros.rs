pub type Result<T> = core::result::Result<T, BLiteError>;

#[derive(Debug)]
pub enum BLiteError {
    // allocator errors
    FailedToAllocateMemory,

    // micro array errors
    NotMatchSize,

    // micro graph errors
    FailedToCreateGraph,
    NotFoundTensor,
    NotFoundBufferData,
    MissingRegistration,
    NotFoundRegistration,
    NotFoundSubgraphs,
    NotFoundBuffers,
    NotFoundOperators,
    NotFoundOperatorCodes,

    // micro operator resolver
    NotFoundOperator(i32),
    OpIndexOutOfBound,

    // micro builtint options
    NotCompatibleOption,

    // micro fully connected
    NotInitializeActivation,
    NotFoundOption,
}
