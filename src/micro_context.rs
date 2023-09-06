use core::fmt::Debug;
use std::marker::PhantomData;

use crate::micro_allocator::{ArenaAllocator, BumpArenaAllocator};

#[derive(Debug)]
pub struct BLiteContext {}

impl BLiteContext {
    pub fn new() -> Self {
        Self {}
    }
}
