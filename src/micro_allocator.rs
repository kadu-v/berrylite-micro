use crate::micro_errors::{BLiteError::*, Result};

pub trait ArenaAllocator {
    unsafe fn alloc(&mut self, size: usize, align: usize) -> Result<*mut u8>;
    unsafe fn alloc_from_offset(
        &mut self,
        offset: usize,
        size: usize,
        align: usize,
    ) -> Result<*mut u8>;
    unsafe fn dealloc(&mut self, ptr: *mut u8, size: usize, align: usize);
    unsafe fn update_offset(&mut self, offset: usize);
    fn print_description(&self);
    fn description(&self) -> Result<(usize, usize)>;
}

#[derive(Debug)]
pub struct BumpArenaAllocator {
    #[allow(dead_code)]
    arena_start: usize,
    arena_end: usize,
    arena_size: usize,
    next: usize,
}

impl BumpArenaAllocator {
    pub unsafe fn new(arena: &'static mut [u8]) -> Self {
        let arena_start = arena.as_ptr() as usize;
        let arena_size = arena.len();
        let arena_end = arena_start + arena_size;
        Self {
            arena_start,
            arena_end,
            arena_size,
            next: arena_start,
        }
    }

    #[inline(always)]
    fn align_up(addr: usize, align: usize) -> usize {
        (addr + align - 1) & !(align - 1)
    }
}

impl ArenaAllocator for BumpArenaAllocator {
    unsafe fn alloc(&mut self, size: usize, align: usize) -> Result<*mut u8> {
        let alloc_size = size;
        let alloc_start = Self::align_up(self.next, align);
        println!("address x: {:x} {}", alloc_start, alloc_size);
        let alloc_next = match alloc_start.checked_add(alloc_size) {
            Some(next) => next,
            None => return Err(FailedToAllocateMemory),
        };
        if alloc_next > self.arena_end {
            Err(FailedToAllocateMemory)
        } else {
            self.next = alloc_next;
            Ok(alloc_start as *mut u8)
        }
    }

    // Must update alloc_next after allocating all tensors
    unsafe fn alloc_from_offset(
        &mut self,
        offset: usize,
        size: usize,
        align: usize,
    ) -> Result<*mut u8> {
        let alloc_size = size;
        let alloc_start = Self::align_up(self.next + offset, align);
        println!("address: {:x} {}", alloc_start, alloc_size);
        let alloc_next = match alloc_start.checked_add(alloc_size) {
            Some(next) => next,
            None => return Err(FailedToAllocateMemory),
        };
        if alloc_next > self.arena_end {
            Err(FailedToAllocateMemory)
        } else {
            Ok(alloc_start as *mut u8)
        }
    }

    unsafe fn update_offset(&mut self, offset: usize) {
        self.next = self.next + offset;
    }

    unsafe fn dealloc(&mut self, _ptr: *mut u8, _size: usize, _align: usize) {
        todo!()
    }

    fn print_description(&self) {
        dbg!(self);
    }

    fn description(&self) -> Result<(usize, usize)> {
        Ok((self.arena_size, self.next - self.arena_start))
    }
}
