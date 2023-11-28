use crate::micro_errors::{BLiteError, Result};
use crate::{micro_allocator::ArenaAllocator, micro_slice::alloc_array_mut};

/*-----------------------------------------------------------------------------*/
/* Struct for an AllocationInfo                                                */
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
