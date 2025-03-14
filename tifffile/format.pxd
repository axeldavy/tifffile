#distutils: language=c++

from libc.stdint cimport uint8_t, int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
from libcpp.vector cimport vector


"""TIFF byte order."""
cdef enum ByteOrder:
    II # Little-endian.
    MM # Big-endian.


cdef union TagValueUnion:
    int64_t i
    double d


cdef struct TagHeader:
    int16_t code
    int16_t datatype
    int32_t count
    uint64_t value        # Raw uninterpreted value from tag
    uint64_t as_offset    # Value interpreted as an offset
    TagValueUnion as_values[8]  # Array of values as int64_t or double


cdef class TiffFormat:
    """TIFF format properties."""

    cdef public int64_t version
    """Version of TIFF header."""

    cdef public ByteOrder byteorder
    """Byteorder of TIFF header."""

    cdef public int64_t offsetsize
    """Size of offsets."""

    cdef public int64_t headersize
    """Size of the format header."""

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

    cpdef bint is_bigtiff(self)
    cpdef bint is_ndpi(self)

    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil
    """Parse TIFF tag header and return code, dtype, count, and value."""

cdef class TiffFormatClassicLE(TiffFormat):
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil

cdef class TiffFormatClassicBE(TiffFormat):
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil

cdef class TiffFormatBigLE(TiffFormat):
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil

cdef class TiffFormatBigBE(TiffFormat):
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil

cdef class TiffFormatNDPI_LE(TiffFormat):
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil