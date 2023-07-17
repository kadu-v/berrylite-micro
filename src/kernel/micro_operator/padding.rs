pub(super) fn compute_padding_height_width(
    padding: usize,
    stride_h: i32,
    stride_w: i32,
    dilation_h_factor: i32,
    dilation_w_factor: i32,
    input_h: i32,
    input_w: i32,
    filter_h: i32,
    filter_w: i32,
    output_h: i32,
    output_w: i32,
) -> (i32, i32, i32, i32) {
    let out_height = compute_out_size(
        padding,
        input_h,
        filter_h,
        stride_h,
        dilation_h_factor,
    );
    let out_width = compute_out_size(
        padding,
        input_w,
        filter_w,
        stride_w,
        dilation_w_factor,
    );
    let (pad_height, offset_h) =
        compute_padding_with_offset(
            stride_h,
            dilation_h_factor,
            input_h,
            filter_h,
            out_height,
        );
    let (pad_width, offset_w) = compute_padding_with_offset(
        stride_w,
        dilation_w_factor,
        input_w,
        filter_w,
        out_width,
    );
    (pad_height, offset_h, pad_width, offset_w)
}

pub(super) fn compute_out_size(
    padding: usize,
    image_size: i32,
    filter_size: i32,
    stride: i32,
    dilation_rate: i32,
) -> i32 {
    let effective_filter_size =
        (filter_size - 1) * dilation_rate + 1;

    if stride == 0 {
        return 0;
    }

    // padding 0: same, 1: valid
    if padding == 0 {
        (image_size + stride - 1) / stride
    } else if padding == 1 {
        (image_size + stride - effective_filter_size)
            / stride
    } else {
        0
    }
}

pub(super) fn compute_padding_with_offset(
    stride: i32,
    dilation_rate: i32,
    input_size: i32,
    filter_size: i32,
    out_size: i32,
) -> (i32, i32) {
    let effective_filter_size =
        (filter_size - 1) * dilation_rate + 1;
    let mut total_padding = (out_size - 1) * stride
        + effective_filter_size
        - input_size;
    total_padding =
        if total_padding > 0 { total_padding } else { 0 };
    let offset = total_padding % 2;
    let pad = total_padding / 2;
    return (pad, offset);
}
