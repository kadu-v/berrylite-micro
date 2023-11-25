use crate::micro_errors::{BLiteError, Result};
use crate::{micro_allocator::ArenaAllocator, micro_slice::alloc_array_mut};

/*-----------------------------------------------------------------------------*/
/* Struct for an AllocationInfo                                                */
/*-----------------------------------------------------------------------------*/
#[derive(Debug, Clone, Copy)]
pub struct AllocationInfo {
    pub idx: usize,
    pub size: usize,
    pub first_time_used: Option<usize>,
    pub last_time_used: Option<usize>,
    pub need_allocation: bool,
}

impl AllocationInfo {
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

    pub fn update_first_time_used(&mut self, info: AllocationInfo) {
        if self.first_time_used.is_none() && info.first_time_used.is_some() {
            self.size = info.size;
            self.idx = info.idx;
            self.first_time_used = info.first_time_used;
            self.need_allocation = info.need_allocation
        }
    }

    pub fn update_last_time_used(&mut self, info: AllocationInfo) {
        self.size = info.size;
        self.idx = info.idx;
        self.last_time_used = info.last_time_used;
        self.need_allocation = info.need_allocation
    }
}

impl PartialEq for AllocationInfo {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
    }
}

impl Eq for AllocationInfo {}

impl PartialOrd for AllocationInfo {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.size.partial_cmp(&other.size)
    }
}

impl Ord for AllocationInfo {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.size.cmp(&other.size)
    }
}

/*-----------------------------------------------------------------------------*/
/* Struct for an AllocationInfo                                                */
/*-----------------------------------------------------------------------------*/
#[derive(Debug)]
pub struct AllocationInfoBuilder<'a> {
    pub infos: &'a mut [AllocationInfo],
    pub cur_idx: usize,
}

impl<'a> AllocationInfoBuilder<'a> {
    pub unsafe fn new(allocator: &mut impl ArenaAllocator, size: usize) -> Result<Self> {
        let infos = alloc_array_mut(allocator, size)?;
        for i in 0..infos.len() {
            infos[i] = AllocationInfo::new(0, 0, None, None, false);
        }

        Ok(Self { infos, cur_idx: 0 })
    }

    pub fn add_info(&mut self, info: &AllocationInfo) -> Result<()> {
        if self.cur_idx >= self.infos.len() {
            return Err(BLiteError::InfoIndexOutOfBound);
        }
        self.infos[self.cur_idx] = info.clone();
        self.cur_idx += 1;

        Ok(())
    }

    pub fn update_first_time_used(&mut self, idx: usize, info: AllocationInfo) {
        let mut cur_info = self.infos[idx];
        cur_info.update_first_time_used(info);
        self.infos[idx] = cur_info;
    }

    pub fn update_last_time_used(&mut self, idx: usize, info: AllocationInfo) {
        let mut cur_info = self.infos[idx];
        cur_info.update_last_time_used(info);
        self.infos[idx] = cur_info;
    }

    pub fn in_place_reverse_sort(&mut self) {
        for i in 1..self.infos.len() {
            if self.infos[i - 1] <= self.infos[i] {
                let mut j = i;
                let tmp = self.infos[i].clone();
                loop {
                    self.infos[j] = self.infos[j - 1];
                    j -= 1;
                    if !(j > 0 && self.infos[j - 1] <= tmp) {
                        break;
                    }
                }
                self.infos[j] = tmp;
            }
        }
    }
}
