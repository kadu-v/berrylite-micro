use crate::micro_allocation_info::AllocationInfo;
/*-----------------------------------------------------------------------------*/
/* Struct for a List Entry                                                     */
/*-----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------*/
/* Struct for a GreedyMemoryPlanner                                            */
/*-----------------------------------------------------------------------------*/
#[derive(Debug)]
pub struct GreedyMemoryPlanner {}

impl GreedyMemoryPlanner {
    pub fn new() -> Self {
        Self {}
    }

    pub fn create_plan(&self, allocation_info: &mut [AllocationInfo]) {}

    fn insert() {}

    fn does_entry_overlap_in_time() {}

    fn next_simultaneous_active_buffer() {}

    fn calculate_offsets_if_needed() {}
}
