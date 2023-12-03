pub mod greedy_memory_planner;

use crate::micro_allocator::ArenaAllocator;
use crate::micro_errors::Result;

pub trait MemoryPlanner {
    fn commit_memory_plan(&mut self, allocator: &mut impl ArenaAllocator) -> Result<()>;
}
