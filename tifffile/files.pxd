# distutils: language=c++
from cpython.buffer cimport Py_buffer

from libc.stdint cimport int64_t, uint64_t
from .utils cimport recursive_mutex

cdef class FileHandle:
    # Private attributes
    cdef recursive_mutex _mutex
    cdef object _file
    cdef object _fh
    cdef str _mode
    cdef str _name
    cdef str _dir
    cdef int64_t _global_offset
    cdef int64_t _current_offset
    cdef int64_t _actual_position
    cdef int64_t _size
    cdef bint _close
    
    cdef object _read_cache
    cdef Py_ssize_t _max_cache_len
    cdef Py_ssize_t _chunk_size

    # Private methods
    cdef int64_t _seek(self, int64_t offset, int64_t whence)
    cdef bytes _actual_read(self, int64_t offset, int64_t size)
    cdef bytes _read(self, int64_t offset, int64_t size)
    
    # Public methods
    cpdef void open(self)
    cpdef void close(self)
    cpdef int64_t fileno(self)
    cpdef bint writable(self)
    cpdef bint seekable(self)
    cpdef int64_t tell(self)
    cpdef int64_t seek(self, int64_t offset, int64_t whence=*)
    cpdef bytes read(self, int64_t size=*)
    cpdef bytes read_at(self, int64_t offset, int64_t size)
    cpdef int64_t readinto(self, object buffer)
    cpdef int64_t write(self, bytes buffer)
    cpdef void flush(self)
    cpdef object memmap_array(self, object dtype, tuple shape, int64_t offset=*, str mode=*, str order=*)
    cpdef object read_array(self, object dtype, int64_t count=*, int64_t offset=*, object out=*)
    cpdef object read_record(self, object dtype, object shape=*, object byteorder=*)
    cpdef int64_t write_empty(self, int64_t size)
    cpdef int64_t write_array(self, object data, object dtype=*)
