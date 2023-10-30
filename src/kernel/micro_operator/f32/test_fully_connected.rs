use super::fully_connected::OpFullyConnected;
use crate::kernel::utils::{testing::Tensor, types::flat_skip_dims};

fn mk_testcase(input: Tensor<f32>, filter: Tensor<f32>, expected: Tensor<f32>) {
    assert_eq!(
        input.shape.len(),
        2,
        "expected of the length of input shape is 2, but got {}",
        input.shape.len()
    );
    assert_eq!(
        filter.shape.len(),
        2,
        "expected of the length of filter shape is 2, but got {}",
        input.shape.len()
    );
    assert_eq!(
        expected.shape.len(),
        2,
        "expected of the length of output shape is 2, but got {}",
        input.shape.len()
    );

    let output_shape = expected.shape;
    let output_data_size = output_shape.iter().fold(1, |x, acc| x * acc) as usize;
    let mut output_data = vec![0f32; output_data_size];

    let batches = flat_skip_dims(&output_shape, output_shape.len() - 1);
    let output_depth = filter.shape[filter.shape.len() - 2] as i32;
    let accum_depth = filter.shape[filter.shape.len() - 1] as i32;

    OpFullyConnected::kernel(
        &input.data,
        None,
        &filter.data,
        &mut output_data,
        batches,
        output_data_size,
        output_depth,
        accum_depth,
        None,
    )
    .expect("fail to execute fully_connected kernel");
    assert_eq!(output_data, expected.data);
}

// filter * input  = output
// (m, n) * (n, l) = (m, l)
#[test]
fn test_fully_connected_2x3_3x2() {
    let filter_shape = [3, 2];
    let input_shape = [2, 3];

    let input = Tensor::ones(&input_shape);
    let filter = Tensor::ones(&filter_shape);
    let expected = Tensor::from(vec![2., 2., 2., 2., 2., 2., 2., 2., 2.], vec![3, 3]);
    mk_testcase(input, filter, expected);
}

#[test]
fn test_fully_connected_3x3_3x4() {
    let filter_shape = [3, 3];
    let input_shape = [3, 4];

    let filter = Tensor::from(
        vec![0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
        vec![3, 3],
    );
    let input = Tensor::from(
        vec![0.0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, 1.6, 2.0, 2.2, 2.4, 2.6],
        vec![3, 4],
    );
    let expected = Tensor::from(vec![2., 2., 2., 2., 2., 2., 2., 2., 2.], vec![3, 3]);
    mk_testcase(input, filter, expected);
}
