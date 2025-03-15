from libc.stdint cimport int64_t, uint16_t
from .pages cimport TiffPage

cpdef object fast_parsing_zstd_delta(
    TiffPage page,
    int64_t start_x,
    int64_t stop_x,
    int64_t start_y,
    int64_t stop_y)
