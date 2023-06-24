use core::mem::size_of;

pub struct ArenaAllocator {
    arena: &'static mut [u8],
    next: usize,
}

impl ArenaAllocator {
    pub unsafe fn new(arena: &'static mut [u8]) -> Self {
        Self { arena, next: 0 }
    }

    pub unsafe fn alloc<T>(&mut self, size: usize) -> Option<*mut u8> {
        let alloc_size = size_of::<T>() * size;
        let next = self.next + alloc_size;
        if self.next >= self.arena.len() || next >= self.arena.len() {
            return None;
        }
        let ptr = self.arena[self.next..next].as_mut_ptr();
        self.next = next;
        return Some(ptr);
    }
}
