
#distutils: language=c++

from libc.stdint cimport int32_t, int64_t
from .files cimport FileHandle
from .format cimport TiffFormat



cdef class TiffTag:
    """TIFF tag structure.

    TiffTag instances are not thread-safe. All attributes are read-only.

    Parameters:
        parent:
            TIFF file tag belongs to.
        offset:
            Position of tag structure in file.
        code:
            Decimal code of tag.
        dtype:
            Data type of tag value item.
        count:
            Number of items in tag value.
        valueoffset:
            Position of tag value in file.

    """

    cdef public FileHandle fh
    """TIFF file as memory-mapped bytes."""
    cdef public TiffFormat tiff_format
    """TIFF format to interprete the tag belongs."""
    cdef public int64_t offset
    """Position of tag structure in file."""
    cdef public int64_t code
    """Decimal code of tag."""
    cdef public int64_t datatype
    """:py:class:`DATATYPE` of tag value item."""
    cdef public int64_t count
    """Number of items in tag value."""
    cdef public int64_t valueoffset
    """Position of tag value in file."""
    cdef object _value

    @staticmethod
    cdef object _read_value(
        FileHandle fh,
        TiffFormat tiff_format,
        int64_t offset,
        int64_t code,
        int64_t datatype,
        int64_t count,
        int64_t valueoffset
    )

    @staticmethod
    cdef object _process_value(
        object value, int64_t code, int64_t datatype, int64_t offset
    )


cdef class TiffTags:

    cdef dict _dict#: dict[int, TiffTag]
    cdef list _list#: list[dict[int, TiffTag]]

    cpdef void add(self, TiffTag tag)

    cpdef list keys(self)
    @staticmethod
    cdef int _offset_key(self, TiffTag tag)

    cpdef list values(self)

    @staticmethod
    cdef int _offset_key2(self, tuple element)

    cpdef list items(self)

    cpdef object valueof(
        self,
        object key,
        object default = ?,
        int64_t index = ?)

    cpdef TiffTag get(
        self,
        object key,
        TiffTag default = ?,
        int64_t index = ?)

    cpdef object getall(
        self,
        object key,
        object default = ?)


cdef class TiffTagRegistry:
    cdef dict _dict#: dict[int | str, str | int]
    cdef list _list#: list[dict[int | str, str | int]]

    cpdef void update(self, object arg)
    cpdef void add(self, int64_t code, str name)

    @staticmethod
    cdef int _code_key(self, tuple element)
    cpdef list items(self)
    cpdef object get(self, object key, object default=?)
    cpdef object getall(self, object key, object default=?)