#![cfg_attr(not(feature = "std"), no_std)]
pub mod kernels;

pub mod builtin_op_data;
pub mod micro_graph;
pub mod tflite_schema_generated;
