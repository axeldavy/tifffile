# distutils: language=c++
from cpython.buffer cimport Py_buffer
from libc.stdio cimport FILE, fopen, fclose, fread, fseek, ftell, setvbuf, SEEK_SET, SEEK_CUR, SEEK_END

from libc.stdint cimport int64_t, uint64_t
from .utils cimport recursive_mutex

cdef class FileHandle:
    # Private attributes
    cdef recursive_mutex _mutex
    cdef str _name
    cdef str _dir
    cdef int64_t _global_offset
    cdef int64_t _size
    cdef int64_t _current_pos  # Track current file position
    
    # C file handling
    cdef FILE* _cfh
    cdef bint _close
    
    # Public methods
    cpdef void open(self)
    cpdef void close(self)
    cpdef int64_t tell(self)
    cdef int64_t tell_c(self) noexcept nogil
    cdef int64_t size_c(self) noexcept nogil
    cpdef int64_t seek(self, int64_t offset, int64_t whence=?)
    cpdef bytes read(self, int64_t size=?)
    cpdef bytes read_at(self, int64_t offset, int64_t size)
    cpdef object read_array(self, object dtype, int64_t count=?, int64_t offset=?, object out=?)
    cpdef object read_record(self, object dtype, object shape=?, object byteorder=?)
    
    # New method with proper locking (changed from cdef to cpdef)
    cdef int64_t read_into(self, unsigned char* dst, int64_t offset, int64_t size) noexcept nogil
