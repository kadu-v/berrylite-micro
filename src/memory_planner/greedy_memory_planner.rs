use crate::{
    micro_allocator::ArenaAllocator,
    micro_array::ArrayElem,
    micro_errors::BLiteError,
    micro_errors::Result,
    micro_graph::TFLiteSubGraph,
    micro_slice::{alloc_array_from_offset, alloc_array_mut},
    micro_tensor::BLiteTensor,
};
use core::mem::size_of;

use super::MemoryPlanner;

/*-----------------------------------------------------------------------------*/
/* Struct for an Requirement                                                   */
/*-----------------------------------------------------------------------------*/
#[derive(Debug, Clone, Copy)]
pub struct Requirement {
    pub idx: usize,
    pub size: usize,
    pub first_time_used: Option<usize>,
    pub last_time_used: Option<usize>,
    pub need_allocation: bool,
}

impl Requirement {
    pub fn new(
        size: usize,
        idx: usize,
        first_time_used: Option<usize>,
        last_time_used: Option<usize>,
        need_allocation: bool,
    ) -> Self {
        Self {
            idx,
            size,
            first_time_used,
            last_time_used,
            need_allocation,
        }
    }

    pub fn update_first_time_used(&mut self, req: Requirement) {
        if self.first_time_used.is_none() && req.first_time_used.is_some() {
            self.size = req.size;
            self.idx = req.idx;
            self.first_time_used = req.first_time_used;
            self.need_allocation = req.need_allocation
        }
    }

    pub fn update_last_time_used(&mut self, req: Requirement) {
        self.size = req.size;
        self.idx = req.idx;
        self.last_time_used = req.last_time_used;
        self.need_allocation = req.need_allocation
    }
}

impl PartialEq for Requirement {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
    }
}

impl Eq for Requirement {}

impl PartialOrd for Requirement {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.size.partial_cmp(&other.size)
    }
}

impl Ord for Requirement {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.size.cmp(&other.size)
    }
}

/*-----------------------------------------------------------------------------*/
/* Struct for an AllocationInfo                                                */
/*-----------------------------------------------------------------------------*/
#[derive(Debug)]
pub struct AllocationInfo<'a> {
    pub info: &'a mut [Requirement],
    pub cur_idx: usize,
}

impl<'a> AllocationInfo<'a> {
    pub unsafe fn new(allocator: &mut impl ArenaAllocator, size: usize) -> Result<Self> {
        let info = alloc_array_mut(allocator, size)?;
        for i in 0..info.len() {
            info[i] = Requirement::new(0, 0, None, None, false);
        }

        Ok(Self { info, cur_idx: 0 })
    }

    pub fn add_info(&mut self, req: &Requirement) -> Result<()> {
        if self.cur_idx >= self.info.len() {
            return Err(BLiteError::InfoIndexOutOfBound);
        }
        self.info[self.cur_idx] = req.clone();
        self.cur_idx += 1;

        Ok(())
    }

    pub fn update_first_time_used(&mut self, idx: usize, req: Requirement) {
        let mut cur_info = self.info[idx];
        cur_info.update_first_time_used(req);
        self.info[idx] = cur_info;
    }

    pub fn update_last_time_used(&mut self, idx: usize, req: Requirement) {
        let mut cur_info = self.info[idx];
        cur_info.update_last_time_used(req);
        self.info[idx] = cur_info;
    }

    pub fn in_place_reverse_sort(&mut self) {
        for i in 1..self.info.len() {
            if self.info[i - 1] <= self.info[i] {
                let mut j = i;
                let tmp = self.info[i].clone();
                loop {
                    self.info[j] = self.info[j - 1];
                    j -= 1;
                    if !(j > 0 && self.info[j - 1] <= tmp) {
                        break;
                    }
                }
                self.info[j] = tmp;
            }
        }
    }
}

/*-----------------------------------------------------------------------------*/
/* Struct for a List Entry                                                     */
/*-----------------------------------------------------------------------------*/
#[derive(Debug, Clone, Copy)]
struct ListEntry {
    pub offset: usize,
    pub requirement_idx: Option<usize>,
    pub next_entry_idx: Option<usize>,
}

impl ListEntry {
    pub fn new(
        offset: usize,
        requirement_idx: Option<usize>,
        next_entry_idx: Option<usize>,
    ) -> Self {
        Self {
            offset,
            requirement_idx,
            next_entry_idx,
        }
    }
}

#[derive(Debug)]
struct OffsetList<'a> {
    list: &'a mut [ListEntry],
    size: usize,
    next_free_idx: usize,
    first_entry_idx: Option<usize>,
}

impl<'a> OffsetList<'a> {
    pub fn new(allocator: &mut impl ArenaAllocator, size: usize) -> Result<Self> {
        let list = unsafe { alloc_array_mut(allocator, size) }?;
        Ok(Self {
            list,
            size,
            next_free_idx: 0,
            first_entry_idx: None,
        })
    }

    pub fn get_first_entry(&self) -> Option<ListEntry> {
        if let Some(first_entry_idx) = self.first_entry_idx {
            return Some(self.list[first_entry_idx]);
        }
        return None;
    }

    pub fn insert_entry(&mut self, entry: ListEntry) -> Result<()> {
        let entry_idx = self.add_entry(entry)?;
        let entry_offset = entry.offset;

        if let Some(first_entry_idx) = self.first_entry_idx {
            let first_entry = self.list[first_entry_idx];
            if first_entry.offset > entry_offset {
                self.list[first_entry_idx].next_entry_idx = Some(first_entry_idx);
                self.first_entry_idx = Some(entry_idx);
            } else {
                let mut cur_entry_idx = first_entry_idx;
                loop {
                    if let Some(next_entry_idx) = self.list[cur_entry_idx].next_entry_idx {
                        let next_entry = self.list[next_entry_idx];
                        if next_entry.offset > entry_offset {
                            self.list[cur_entry_idx].next_entry_idx = Some(entry_idx);
                            self.list[entry_idx].next_entry_idx = Some(next_entry_idx);
                            break;
                        }
                        cur_entry_idx = next_entry_idx;
                    } else {
                        self.list[cur_entry_idx].next_entry_idx = Some(entry_idx);
                        self.list[entry_idx].next_entry_idx = None;
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.list.len()
    }

    fn add_entry(&mut self, entry: ListEntry) -> Result<usize> {
        if self.next_free_idx + 1 > self.size {
            return Err(BLiteError::OutOfListEntrySize);
        }
        if self.first_entry_idx.is_none() {
            self.first_entry_idx = Some(self.next_free_idx);
        }
        let idx = self.next_free_idx;
        self.next_free_idx += 1;
        self.list[idx] = entry;

        return Ok(idx);
    }
}

/*-----------------------------------------------------------------------------*/
/* Struct for a GreedyMemoryPlanner                                            */
/*-----------------------------------------------------------------------------*/
#[derive(Debug)]
pub struct GreedyMemoryPlanner<'a, 'b, 'c, 'd, T: ArrayElem<T>> {
    allocation_info: AllocationInfo<'a>,
    offset_list: OffsetList<'b>,
    subgraph: &'d TFLiteSubGraph<'c>,
    tensors: &'c mut [BLiteTensor<'c, T>],
}

impl<'a, 'b, 'c, 'd, T: ArrayElem<T>> GreedyMemoryPlanner<'a, 'b, 'c, 'd, T> {
    pub fn new(
        allocator: &mut impl ArenaAllocator,
        subgraph: &'d TFLiteSubGraph<'c>,
        tensors: &'c mut [BLiteTensor<'c, T>],
    ) -> Result<Self> {
        let dummy_allocation_info = unsafe { AllocationInfo::new(allocator, 0) }?;
        let dummy_offset_list = OffsetList::new(allocator, 0)?;
        Ok(Self {
            allocation_info: dummy_allocation_info,
            offset_list: dummy_offset_list,
            subgraph,
            tensors,
        })
    }

    fn commit_memory_plan(&mut self, allocator: &mut impl ArenaAllocator) -> Result<()> {
        self.allocate_inputs_outputs(allocator)?;
        let allocation_info = self.calculate_allocation_info(allocator)?;
        let offset_list = OffsetList::new(allocator, allocation_info.info.len())?;
        self.allocation_info = allocation_info;
        self.offset_list = offset_list;
        self.allocate_intermediate_tensors(allocator)?;
        Ok(())
    }

    fn calculate_allocation_info(
        &self,
        allocator: &mut impl ArenaAllocator,
    ) -> Result<AllocationInfo<'a>> {
        // TODO: should be drop this all allocation infos after creating allocation infos
        let mut all_alloc_info = unsafe { AllocationInfo::new(allocator, self.tensors.len()) }?;
        for (time_step, op) in self.subgraph.operators().unwrap().iter().enumerate() {
            let inputs = op.inputs().unwrap();
            let outputs = op.outputs().unwrap();

            // check last_time_used using inputs
            for idx in inputs {
                let idx = idx as usize;
                let size = self.tensors[idx].size();
                let last_time_used = time_step;
                let need_allocation = self.tensors[idx].len() == 0;
                let info = Requirement::new(size, idx, None, Some(last_time_used), need_allocation);
                all_alloc_info.update_last_time_used(idx, info);
            }

            // check first_time_used using outputs
            for idx in outputs {
                let idx = idx as usize;
                let size = self.tensors[idx].size();
                let first_time_used = time_step;
                let need_allocation = self.tensors[idx].len() == 0;
                let info =
                    Requirement::new(size, idx, Some(first_time_used), None, need_allocation);
                all_alloc_info.update_first_time_used(idx, info);
            }
        }

        let need_allocation_count =
            all_alloc_info.info.iter().fold(
                0,
                |acc, &info| {
                    if info.need_allocation {
                        acc + 1
                    } else {
                        acc
                    }
                },
            );

        let mut allocation_info = unsafe { AllocationInfo::new(allocator, need_allocation_count) }?;
        for info in all_alloc_info.info.iter() {
            if info.need_allocation {
                allocation_info.add_info(info)?;
            }
        }

        allocation_info.in_place_reverse_sort();

        Ok(allocation_info)
    }

    fn allocate_intermediate_tensors(&mut self, allocator: &mut impl ArenaAllocator) -> Result<()> {
        self.calculate_offsets_if_needed()?;

        // allocate tensors following memory plan
        unsafe { self.allocate_tensors_following_plan(allocator) }
    }

    fn allocate_inputs_outputs(&mut self, allocator: &mut impl ArenaAllocator) -> Result<()> {
        let inputs = self.subgraph.inputs().unwrap();
        let outputs = self.subgraph.outputs().unwrap();

        for input_idx in inputs.iter() {
            let size = self.tensors[input_idx as usize].size();
            let data = unsafe { alloc_array_mut(allocator, size)? };
            let mut tensor = self.tensors[input_idx as usize]._t()?.borrow_mut();
            tensor.data = data;
        }

        for output_idx in outputs.iter() {
            let size = self.tensors[output_idx as usize].size();
            let data = unsafe { alloc_array_mut(allocator, size)? };
            let mut tensor = self.tensors[output_idx as usize]._t()?.borrow_mut();
            tensor.data = data;
        }
        Ok(())
    }

    fn does_entry_overlap_in_time(
        &self,
        entry: ListEntry,
        first_time_used: usize,
        last_time_used: usize,
    ) -> Result<bool> {
        if let Some(requirement_idx) = entry.requirement_idx {
            let req = self.allocation_info.info[requirement_idx];
            if req.first_time_used.unwrap() > last_time_used {
                return Ok(false);
            }
            if first_time_used > req.last_time_used.unwrap() {
                return Ok(false);
            }
        } else {
            return Err(BLiteError::NotFoundRequirementIdx);
        }
        Ok(true)
    }

    pub unsafe fn allocate_tensors_following_plan(
        &self,
        allocator: &mut impl ArenaAllocator,
    ) -> Result<()> {
        let mut max_offset = 0;
        for entry in self.offset_list.list.iter() {
            let offset = entry.offset;
            let requirement_idx = entry.requirement_idx.unwrap();
            let req = self.allocation_info.info[requirement_idx];
            let tensor_idx = req.idx;
            let size = self.tensors[tensor_idx].size();
            let data = unsafe { alloc_array_from_offset::<T>(allocator, offset, size) }?;
            self.tensors[tensor_idx]._t()?.borrow_mut().data = data;

            if max_offset < offset + size {
                max_offset = offset + size;
            }
        }
        allocator.update_offset(size_of::<T>() * max_offset)
    }

    fn next_simultaneous_active_buffer(
        &self,
        start: Option<ListEntry>,
        first_time_used: usize,
        last_time_used: usize,
    ) -> Result<Option<ListEntry>> {
        let mut result = None;
        let mut candidate_next_entry;

        if let Some(start_entry) = start {
            if let Some(next_entry_idx) = start_entry.next_entry_idx {
                candidate_next_entry = self.offset_list.list[next_entry_idx];
            } else {
                return Ok(result);
            }
        } else {
            candidate_next_entry = self.offset_list.get_first_entry().unwrap();
        }

        loop {
            if self.does_entry_overlap_in_time(
                candidate_next_entry,
                first_time_used,
                last_time_used,
            )? {
                result = Some(candidate_next_entry);
                break;
            }

            if let Some(next_entry_idx) = candidate_next_entry.next_entry_idx {
                candidate_next_entry = self.offset_list.list[next_entry_idx];
            } else {
                break;
            }
        }

        return Ok(result);
    }

    fn add_entry(&mut self, entry: ListEntry) -> Result<usize> {
        self.offset_list.add_entry(entry)
    }

    fn insert_entry(&mut self, entry: ListEntry) -> Result<()> {
        self.offset_list.insert_entry(entry)
    }

    pub fn calculate_offsets_if_needed(&mut self) -> Result<()> {
        // add first entry to offset list
        let buffer_offset = 0;
        let requirement_idx = 0;
        let first_entry = ListEntry::new(buffer_offset, Some(requirement_idx), None);
        let first_entry_idx = self.add_entry(first_entry)?;

        for i in 1..self.offset_list.len() {
            let buffer_id = i;
            let wanted_requirement = self.allocation_info.info[i];
            let wanted_size = wanted_requirement.size;
            let wanted_first_time_used = wanted_requirement.first_time_used.unwrap();
            let wanted_last_time_used = wanted_requirement.last_time_used.unwrap();

            let mut candidate_offset = first_entry_idx;
            let mut prior_entry = None;
            loop {
                let next_entry = self.next_simultaneous_active_buffer(
                    prior_entry,
                    wanted_first_time_used,
                    wanted_last_time_used,
                )?;

                if let Some(prior_entry) = prior_entry {
                    let candidate_requirement =
                        self.allocation_info.info[prior_entry.requirement_idx.unwrap()];
                    let entry_offset = prior_entry.offset + candidate_requirement.size;
                    if entry_offset > candidate_offset {
                        candidate_offset = entry_offset;
                    }
                }
                if let Some(next_entry) = next_entry {
                    let gap = next_entry.offset as i32 - candidate_offset as i32;
                    if gap >= wanted_size as i32 {
                        break;
                    }
                } else {
                    break;
                }
                prior_entry = next_entry;
            }

            // add this entry to list
            let new_entry_offset = candidate_offset;
            let new_entry_requirement_idx = Some(buffer_id);
            let new_entry = ListEntry::new(new_entry_offset, new_entry_requirement_idx, None);
            self.insert_entry(new_entry)?;
        }
        Ok(())
    }
}

impl<'a, 'b, 'c, 'd, T: ArrayElem<T>> MemoryPlanner for GreedyMemoryPlanner<'a, 'b, 'c, 'd, T> {
    fn commit_memory_plan(&mut self, allocator: &mut impl ArenaAllocator) -> Result<()> {
        self.commit_memory_plan(allocator)?;
        Ok(())
    }
}
