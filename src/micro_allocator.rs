use crate::micro_erros::{BLiteError::*, Result};

pub trait ArenaAllocator {
    unsafe fn alloc(
        &mut self,
        size: usize,
        align: usize,
    ) -> Result<*mut u8>;
    unsafe fn dealloc(
        &mut self,
        ptr: *mut u8,
        size: usize,
        align: usize,
    );
}
pub struct BumpArenaAllocator {
    arena: &'static mut [u8],
    next: usize,
}

impl BumpArenaAllocator {
    pub unsafe fn new(arena: &'static mut [u8]) -> Self {
        Self { arena, next: 0 }
    }

    fn align_up(addr: usize, align: usize) -> usize {
        (addr + align - 1) & !(align - 1)
    }
}

impl ArenaAllocator for BumpArenaAllocator {
    unsafe fn alloc(
        &mut self,
        size: usize,
        align: usize,
    ) -> Result<*mut u8> {
        let alloc_size = size;
        let alloc_start = Self::align_up(self.next, align);
        let alloc_next =
            match alloc_start.checked_add(alloc_size) {
                Some(next) => next,
                None => return Err(FaildToAllocateMemory),
            };

        if alloc_next > self.arena.len() {
            Err(FaildToAllocateMemory)
        } else {
            let ptr = self.arena[self.next..alloc_next]
                .as_mut_ptr();
            self.next = alloc_next;
            Ok(ptr)
        }
    }

    unsafe fn dealloc(
        &mut self,
        ptr: *mut u8,
        size: usize,
        align: usize,
    ) {
        todo!()
    }
}
