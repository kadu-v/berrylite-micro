#[derive(Debug, Clone, Copy)]
pub struct BufferRequirement {
    pub size: usize,
    pub first_time_used: usize,
    pub last_time_used: usize,
}

impl BufferRequirement {
    pub fn new(size: usize, first_time_used: usize, last_time_used: usize) -> Self {
        Self {
            size,
            first_time_used,
            last_time_used,
        }
    }
}

#[derive(Debug)]
pub struct GreedyMemoryPlanner {}
