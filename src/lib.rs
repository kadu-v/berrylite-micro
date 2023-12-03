#![feature(trait_alias)]
#![feature(core_intrinsics)]
#![cfg_attr(feature = "no_std", no_std)]
pub mod kernel;
pub(crate) mod memory_planner;
pub mod micro_allocator;
pub mod micro_array;
pub(crate) mod micro_context;
pub mod micro_errors;
pub(crate) mod micro_graph;
pub mod micro_interpreter;
pub(crate) mod micro_node;
pub mod micro_op_resolver;
pub(crate) mod micro_registration;
pub(crate) mod micro_slice;
pub(crate) mod micro_tensor;
pub mod tflite_schema_generated;
