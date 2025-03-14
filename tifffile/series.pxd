#distutils: language=c++

from libc.stdint cimport int64_t
from libcpp.vector cimport vector

from .files cimport FileHandle
from .format cimport TiffFormat
from .pages cimport TiffPage

cdef class TiffPages:
    """Sequence of TIFF image file directories (IFD chain)."""
    
    # Public attributes
    cdef readonly FileHandle filehandle
    cdef readonly TiffFormat tiff
    
    # Private attributes - core data structures
    cdef list _pages  # List containing TiffPage or TiffFrame
    cdef vector[int64_t] _page_offsets     # Offsets of each page

    @staticmethod
    cdef TiffPages from_file(
        FileHandle filehandle,
        TiffFormat tiff,
        int64_t offset
    )
    @staticmethod
    cdef TiffPages from_parent(
        FileHandle filehandle,
        TiffFormat tiff,
        vector[int64_t] offsets
    )
    cdef TiffPage getpage(self, int64_t key)
