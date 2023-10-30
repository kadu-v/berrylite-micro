pub fn flat_skip_dims(shape: &[i32], skip_dim: usize) -> i32 {
    let mut flat_size = 1;
    for (i, &e) in shape.iter().enumerate() {
        if i != skip_dim {
            flat_size *= e;
        }
    }
    flat_size
}
