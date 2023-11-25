pub type Result<T> = core::result::Result<T, BLiteError>;

#[derive(Debug)]
pub enum BLiteError {
    //
    InfoIndexOutOfBound,

    // allocator errors
    FailedToAllocateMemory,

    // micro array errors
    NotMatchSize,

    // micro tensors
    NotBTensor,
    NotI32Tensor,

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
    InCompatibleType,

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
    NotFoundQuantParams,

    // micro reshape
    InCompatibleShape(i32, i32),
    // micro_activation
    NotFoundFusedActivation(i32),
    FatalError,
}
