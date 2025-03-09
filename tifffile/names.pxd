# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# distutils: language=c++
from libc.stdint cimport int32_t

cdef object get_tag_names(int32_t tag)