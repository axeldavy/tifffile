#distutils: language=c++

from libc.stdint cimport uint8_t, int32_t, int64_t
from .files cimport FileHandle
from .format cimport TiffFormat, TagHeader
from libcpp.unordered_set cimport unordered_set as cpp_set
from libcpp.unordered_map cimport unordered_multimap as cpp_multimap
from libcpp.string cimport string as cpp_string
from libcpp.vector cimport vector


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
        datatype:
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
    cdef TiffTag fromfile(
        FileHandle fh,
        TiffFormat tiff_format,
        int64_t offset,
        bytes header,
        bint validate
    )

    @staticmethod
    cdef TiffTag fromheader(FileHandle fh,
                            TiffFormat tiff_format,
                            TagHeader header)

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

    cdef object value_get(self)
    cdef void value_set(self, object)


cdef class TiffTags:
    cdef FileHandle fh
    cdef TiffFormat tiff
    cdef list _tags  # Store all tags in a list. None if not loaded into a TiffTag
    cdef vector[TagHeader] _headers # headers for each tag
    cdef cpp_set[int64_t] _code_set
    cdef cpp_multimap[int64_t, int] _code_indices  # Maps code -> list indices
    
    # Private methods
    cdef int64_t _get_tag_value_as_int64(self, int index) noexcept nogil
    cdef double _get_tag_value_as_double(self, int index) noexcept nogil
    cdef bint _get_tag_value_as_int64_vec(self, vector[int64_t]& dst, int index) noexcept nogil
    cdef object _get_tag_value_at(self, int index)
    cdef TiffTag _get_tag_at(self, int index)
    cdef int load_tags(self, uint8_t* data, int64_t count, int64_t size) noexcept nogil
    cpdef list keys(self)
    cpdef list items(self)
    cpdef list values(self)
    cpdef object valueof(
        self,
        int64_t code,
        object default=?,
        int64_t index=?,
    )
    cdef int64_t valueof_int(
        self,
        int64_t code,
        int64_t default,
        int64_t index) noexcept nogil
    cdef double valueof_double(
        self,
        int64_t code,
        double default,
        int64_t index) noexcept nogil
    cdef bint valueof_int_array(
        self,
        vector[int64_t] &dst,
        int64_t code,
        int64_t index) noexcept nogil
    cpdef TiffTag get(
        self,
        int64_t code,
        TiffTag default=?,
        int64_t index=?)
    cdef int64_t get_count(
        self,
        int64_t code,
        int64_t index) noexcept nogil
    cdef bint contains_code(self, int64_t code) noexcept nogil


cdef class TiffTagRegistry:
    cdef list _entries  # Store all entries as (code, name) tuples
    cdef cpp_multimap[int64_t, int] _code_indices  # Maps code -> list index
    cdef cpp_multimap[cpp_string, int] _name_indices # Maps str -> list index
    cdef int64_t _entry_count  # Count of non-None entries
    
    # Fast cdef methods
    cdef bint _contains_code(self, int64_t code) nogil
    cdef bint _contains_name(self, cpp_string name) nogil
    cdef vector[int] _get_code_indices(self, int64_t code) nogil
    cdef vector[int] _get_name_indices(self, cpp_string name) nogil
    cdef void _add_index(self, int64_t code, str name, int index) nogil

    cpdef void update(self, object arg)
    cpdef void add(self, int64_t code, str name)
    @staticmethod
    cdef int64_t _code_key(self, tuple element)
    cpdef list items(self)
    cpdef object get(self, object key, object default=?)
    cpdef object getall(self, object key, object default=?)
    cpdef bint contains(self, object item)