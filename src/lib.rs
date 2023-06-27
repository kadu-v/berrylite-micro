#![cfg_attr(not(feature = "std"), no_std)]
pub mod kernels;

pub mod builtin_op_data;
pub mod micro_allocator;
pub mod micro_array;
pub mod micro_context;
pub mod micro_erros;
pub mod micro_graph;
pub mod micro_op_resolver;
pub mod micro_operator;
pub mod micro_registration;
pub mod micro_slice;
pub mod tflite_schema_generated;
