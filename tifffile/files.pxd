from cpython.buffer cimport Py_buffer
import numpy as np
cimport numpy as np

from libc.stdint cimport int64_t, uint64_t

cdef class FileHandle:
    # Private attributes
    cdef object _file
    cdef object _fh
    cdef str _mode
    cdef str _name
    cdef str _dir
    cdef object _lock
    cdef int64_t _offset
    cdef int64_t _size
    cdef bint _close
    
    cdef object _read_cache
    cdef Py_ssize_t _max_cache_len
    cdef Py_ssize_t _chunk_size

    cdef bytes _read(self, int64_t offset, int64_t size)
    
    # Methods
    cpdef void open(self) except *
    cpdef void close(self)
    cpdef int fileno(self) except -1
    cpdef bint writable(self)
    cpdef bint seekable(self)
    cpdef int64_t tell(self)
    cpdef int64_t seek(self, int64_t offset, int whence=*) except -1
    cpdef bytes read(self, Py_ssize_t size=*)
    cpdef Py_ssize_t readinto(self, object buffer) except -1
    cpdef Py_ssize_t write(self, bytes buffer) except -1
    cpdef void flush(self) except *
    cpdef np.ndarray memmap_array(self, object dtype, tuple shape, int64_t offset=*, str mode=*, str order=*)
    cpdef np.ndarray read_array(self, object dtype, Py_ssize_t count=*, int64_t offset=*, object out=*)
    cpdef object read_record(self, object dtype, object shape=*, object byteorder=*)
    cpdef Py_ssize_t write_empty(self, Py_ssize_t size) except -1
    cpdef Py_ssize_t write_array(self, np.ndarray data, object dtype=*)
    cpdef void set_lock(self, bint value) except *
