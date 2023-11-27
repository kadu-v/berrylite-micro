use core::borrow::BorrowMut;

use crate::{
    micro_allocation_info::AllocationInfo,
    micro_allocator::ArenaAllocator,
    micro_array::ArrayElem,
    micro_errors::BLiteError,
    micro_errors::Result,
    micro_slice::{alloc_array_from_offset, alloc_array_mut},
    micro_tensor::BLiteTensor,
};

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
pub struct GreedyMemoryPlanner<'a, 'b> {
    info: &'b mut AllocationInfo<'b>,
    offset_list: OffsetList<'a>,
}

impl<'a, 'b> GreedyMemoryPlanner<'a, 'b> {
    pub fn new(
        allocator: &mut impl ArenaAllocator,
        info: &'b mut AllocationInfo<'b>,
    ) -> Result<Self> {
        let offset_list = OffsetList::new(allocator, info.info.len())?;
        Ok(Self { info, offset_list })
    }

    fn does_entry_overlap_in_time(
        &self,
        entry: ListEntry,
        first_time_used: usize,
        last_time_used: usize,
    ) -> Result<bool> {
        if let Some(requirement_idx) = entry.requirement_idx {
            let req = self.info.info[requirement_idx];
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

    pub unsafe fn allocate_tensors<T: ArrayElem<T>>(
        &mut self,
        allocator: &mut impl ArenaAllocator,
        tensors: &mut [BLiteTensor<'a, T>],
    ) -> Result<()> {
        let mut max_offset = 0;
        for entry in self.offset_list.list.iter() {
            let offset = entry.offset;
            let requirement_idx = entry.requirement_idx.unwrap();
            let req = self.info.info[requirement_idx];
            let tensor_idx = req.idx;
            let size = tensors[tensor_idx].size();
            let data = unsafe { alloc_array_from_offset::<T>(allocator, offset, size) }?;
            tensors[tensor_idx]._t()?.borrow_mut().data = data;

            if max_offset < offset + size {
                max_offset = offset + size;
            }
        }
        allocator.update_offset(max_offset);
        Ok(())
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
            let wanted_requirement = self.info.info[i];
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
                        self.info.info[prior_entry.requirement_idx.unwrap()];
                    let entry_offset = prior_entry.offset + candidate_requirement.size;
                    if entry_offset > candidate_offset {
                        candidate_offset = entry_offset;
                    }
                }
                if let Some(next_entry) = next_entry {
                    println!("{:?} {}", next_entry, candidate_offset);
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

        for (i, info) in self.info.info.iter().enumerate() {
            println!("-----------------------------------");
            println!("{:?}", self.offset_list.list[i]);
            println!("{:?}", info);
        }
        println!("-----------------------------------");

        Ok(())
    }
}
