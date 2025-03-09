#distutils: language=c++

from libc.stdint cimport int64_t, uint64_t



cdef enum class ByteOrder:
    """TIFF byte order."""
    II # Little-endian.
    MM # Big-endian.

cdef class TiffFormat:
    """TIFF format properties."""

    cdef public int64_t version
    """Version of TIFF header."""

    cdef public ByteOrder byteorder
    """Byteorder of TIFF header."""

    cdef public int64_t offsetsize
    """Size of offsets."""

    cdef public str offsetformat
    """Struct format for offset values."""

    cdef public int64_t tagnosize
    """Size of `tagnoformat`."""

    cdef public str tagnoformat
    """Struct format for number of TIFF tags."""

    cdef public int64_t tagsize
    """Size of `tagformat1` and `tagformat2`."""

    cdef public str tagformat1
    """Struct format for code and dtype of TIFF tag."""

    cdef public str tagformat2
    """Struct format for count and value of TIFF tag."""

    cdef public int64_t tagoffsetthreshold
    """Size of inline tag values."""

    cdef int64_t _hash
    
    #@staticmethod
    #cdef TiffFormat detect_format(bytes header)
    """Detect appropriate TIFF format from file header."""
    
    cpdef tuple parse_tag_header(self, bytes header)
    """Parse TIFF tag header and return code, dtype, count, and value."""
    
    cpdef uint64_t interprete_offset(self, bytes value)
    """Convert bytes value from tag header to offset."""

cdef class TiffFormatClassicLE(TiffFormat):
    cpdef tuple parse_tag_header(self, bytes header)
    cpdef uint64_t interprete_offset(self, bytes value)

cdef class TiffFormatClassicBE(TiffFormat):
    cpdef tuple parse_tag_header(self, bytes header)
    cpdef uint64_t interprete_offset(self, bytes value)

cdef class TiffFormatBigLE(TiffFormat):
    cpdef tuple parse_tag_header(self, bytes header)
    cpdef uint64_t interprete_offset(self, bytes value)

cdef class TiffFormatBigBE(TiffFormat):
    cpdef tuple parse_tag_header(self, bytes header)
    cpdef uint64_t interprete_offset(self, bytes value)

cdef class TiffFormatNDPI_LE(TiffFormat):
    cpdef tuple parse_tag_header(self, bytes header)
    cpdef uint64_t interprete_offset(self, bytes value)