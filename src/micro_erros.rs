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

    // micro builtin options
    NotCompatibleOption,

    // micro fully connected
    NotInitializeActivation,
    NotFoundOption,
    InCompatibleCasting,

    // micro fully connected int8
    NotMatchScale(f64),

    // micro reshape
    InCompatibleShape(i32, i32),
    //
}
