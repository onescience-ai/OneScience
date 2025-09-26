def depth_to_space(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    if ch % (scale_factor * scale_factor) != 0:
        raise ValueError(
            "channel of tensor must be divisible by "
            "(scale_factor * scale_factor) must be an integer multiple."
        )

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    tensor = tensor.permute([0, 1, 4, 2, 5, 3])
    tensor = tensor.reshape(
        [num, new_ch, new_height, new_width])
    return tensor
