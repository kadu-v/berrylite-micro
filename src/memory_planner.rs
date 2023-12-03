pub mod greedy_memory_planner;

use crate::micro_allocator::ArenaAllocator;
use crate::micro_array::ArrayElem;
use crate::micro_errors::Result;
use crate::micro_tensor::BLiteTensor;

pub trait MemoryPlanner<'c> {
    fn commit_memory_plan(&mut self, allocator: &mut impl ArenaAllocator) -> Result<()>;
}
