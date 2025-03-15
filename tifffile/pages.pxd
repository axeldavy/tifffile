#cython: language_level=3
from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from libcpp.vector cimport vector

from .files cimport FileHandle
from .format cimport ByteOrder, TiffFormat
from .tags cimport TiffTags

cdef class TiffPage:
    """TIFF image file directory (IFD)."""
    
    # Public attributes
    cdef public FileHandle fh
    cdef public TiffFormat tiff
    cdef public TiffTags tags
    cdef public int64_t offset
    cdef public tuple shape
    cdef public object dtype
    cdef public str axes
    cdef public vector[int64_t] _dataoffsets
    cdef public vector[int64_t] _databytecounts
    cdef public int64_t[5] shaped
    
    # Internal attributes
    cdef object _dtype
    cdef dict _cache
    
    # Page properties
    cdef public int64_t subfiletype
    cdef public int64_t imagewidth
    cdef public int64_t imagelength
    cdef public int64_t imagedepth
    cdef public int64_t tilewidth
    cdef public int64_t tilelength 
    cdef public int64_t tiledepth
    cdef public int64_t samplesperpixel
    cdef public int64_t bitspersample
    cdef public int64_t sampleformat
    cdef public int64_t rowsperstrip
    cdef public int64_t compression
    cdef public int64_t planarconfig
    cdef public int64_t fillorder
    cdef public int64_t photometric
    cdef public int64_t predictor
    cdef public tuple extrasamples
    cdef public tuple subsampling
    cdef public tuple subifds
    cdef public bytes jpegtables
    cdef public bytes jpegheader
    cdef public float nodata

    @staticmethod
    cdef TiffPage from_file(
        FileHandle filehandle,
        TiffFormat tiff,
        int64_t offset
    )

    # Read IFD structure and tags
    cdef int _read_ifd_structure(self) noexcept nogil
    cdef void _process_common_tags(self) noexcept nogil
    cdef void _process_common_tags_gil(self)
    cdef void _process_special_format_tags(self) noexcept nogil
    cdef void _process_special_format_tags_stk(self)
    cdef void _process_special_format_tags_imagej(self)
    cdef void _process_special_format_tags_bitspersample_2(self)
    cdef void _process_special_format_tags_sampleformat_2(self)
    cdef void _process_special_format_tags_gdal(self)
    cdef int _process_data_pointers(self) noexcept nogil
    cdef void _determine_shape_and_dtype(self)

    # Read image data
    cdef object _asarray_memmap(self)
    cdef object _asarray_contiguous(self, out)
    cdef object _asarray_ndpi_jpeg(self, out)
    cdef object _asarray_tiled(self, out, int64_t maxworkers, int64_t buffersize)
    
    # Property declarations
    #cpdef TiffPage keyframe(self)
    #cpdef int64_t index(self)
    cpdef bint is_contiguous(self)
    cpdef bint is_final(self)
    cpdef bint is_memmappable(self)
    cpdef bint is_tiled(self)
    cpdef bint is_subsampled(self)
    cpdef bint is_jfif(self)
    cpdef int64_t hash(self)
    cpdef bint is_ndpi(self)
    cpdef bint is_philips(self)
    cpdef bint is_eer(self)
    cpdef bint is_mediacy(self)
    cpdef bint is_stk(self)
    cpdef bint is_lsm(self)
    cpdef bint is_fluoview(self)
    cpdef bint is_nih(self)
    cpdef bint is_volumetric(self)
    cpdef bint is_vista(self)
    cpdef bint is_metaseries(self)
    cpdef bint is_ome(self)
    cpdef bint is_scn(self)
    cpdef bint is_micromanager(self)
    cpdef bint is_andor(self)
    cpdef bint is_pilatus(self)
    cpdef bint is_epics(self)
    cpdef bint is_tvips(self)
    cpdef bint is_fei(self)
    cpdef bint is_sem(self)
    cpdef bint is_svs(self)
    cpdef bint is_bif(self)
    cpdef bint is_scanimage(self)
    cpdef bint is_indica(self)
    cpdef bint is_avs(self)
    cpdef bint is_qpi(self)
    cpdef bint is_geotiff(self)
    cpdef bint is_gdal(self)
    cpdef bint is_astrotiff(self)
    cpdef bint is_streak(self)
    cpdef bint is_dng(self)
    cpdef bint is_tiffep(self)
    cpdef bint is_sis(self)
    cpdef bint is_frame(self)
    cpdef bint is_virtual(self)
    cpdef bint is_reduced(self)
    cpdef bint is_multipage(self)
    cpdef bint is_mask(self)
    cpdef bint is_mrc(self)
    cpdef bint is_imagej(self)
    cpdef bint is_shaped(self)
    cpdef bint is_mdgel(self)
    cpdef bint is_agilent(self)