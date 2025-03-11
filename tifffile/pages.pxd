#cython: language_level=3
from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t

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
    cdef public tuple dataoffsets
    cdef public tuple databytecounts
    cdef public int64_t[5] shaped
    
    # Internal attributes
    cdef object _dtype
    cdef tuple _index
    cdef dict _cache
    
    # Page properties
    cdef public int subfiletype
    cdef public int imagewidth
    cdef public int imagelength
    cdef public int imagedepth
    cdef public int tilewidth
    cdef public int tilelength 
    cdef public int tiledepth
    cdef public int samplesperpixel
    cdef public int bitspersample
    cdef public int sampleformat
    cdef public int rowsperstrip
    cdef public int compression
    cdef public int planarconfig
    cdef public int fillorder
    cdef public int photometric
    cdef public int predictor
    cdef public tuple extrasamples
    cdef public tuple subsampling
    cdef public tuple subifds
    cdef public bytes jpegtables
    cdef public bytes jpegheader
    cdef public str software
    cdef public str description
    cdef public str description1
    cdef public float nodata
    
    # Property declarations
    #cpdef TiffPage keyframe(self)
    #cpdef int index(self)
    cpdef bint is_tiled(self)
    cpdef bint is_subsampled(self)
    cpdef bint is_jfif(self)
    cpdef int hash(self)
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
    cpdef bint is_subifd(self)
    cpdef bint is_reduced(self)
    cpdef bint is_multipage(self)
    cpdef bint is_mask(self)
    cpdef bint is_mrc(self)
    cpdef bint is_imagej(self)
    cpdef bint is_shaped(self)
    cpdef bint is_mdgel(self)
    cpdef bint is_agilent(self)

'''
cdef public TiffTags tags
    """Tags belonging to page."""
    cdef public object parent
    """TiffFile instance page belongs to."""
    cdef public int64_t offset
    """Position of page in file."""
    cdef public tuple shape#: tuple[int, ...]
    """Shape of image array in page."""
    cdef public object dtype#: numpy.dtype[Any] | None
    """Data type of image array in page."""
    cdef public int64_t[5] shaped
    """Normalized 5-dimensional shape of image array in page:

        0. separate samplesperpixel or 1.
        1. imagedepth or 1.
        2. imagelength.
        3. imagewidth.
        4. contig samplesperpixel or 1.

    """
    cdef public str axes
    """Character codes for dimensions in image array:
    'S' sample, 'X' width, 'Y' length, 'Z' depth.
    """
    cdef public tuple dataoffsets#: tuple[int, ...]
    """Positions of strips or tiles in file."""
    cdef public tuple databytecounts#: tuple[int, ...]
    """Size of strips or tiles in file."""
    cdef object _dtype#: numpy.dtype[Any] | None
    cdef tuple _index#: tuple[int, ...]  # index of page in IFD tree

    # default properties; might be updated from tags
    cdef public int subfiletype
    """:py:class:`FILETYPE` kind of image."""
    cdef public int imagewidth
    """Number of columns (pixels per row) in image."""
    cdef public int imagelength
    """Number of rows in image."""
    cdef public int imagedepth
    """Number of Z slices in image."""
    cdef public int tilewidth
    """Number of columns in each tile."""
    cdef public int tilelength
    """Number of rows in each tile."""
    cdef public int tiledepth
    """Number of Z slices in each tile."""
    cdef public int samplesperpixel
    """Number of components per pixel."""
    cdef public int bitspersample
    """Number of bits per pixel component."""
    cdef public int sampleformat
    """:py:class:`SAMPLEFORMAT` type of pixel components."""
    cdef public int rowsperstrip
    """Number of rows per strip."""
    cdef public int compression
    """:py:class:`COMPRESSION` scheme used on image data."""
    cdef public int planarconfig
    """:py:class:`PLANARCONFIG` type of storage of components in pixel."""
    cdef public int fillorder
    """Logical order of bits within byte of image data."""
    cdef public int photometric
    """:py:class:`PHOTOMETRIC` color space of image."""
    cdef public int predictor
    """:py:class:`PREDICTOR` applied to image data before compression."""
    cdef public tuple extrasamples # tuple[int, ...]
    """:py:class:`EXTRASAMPLE` interpretation of extra components in pixel."""
    cdef public tuple subsampling # int64_t[2]
    """Subsampling factors used for chrominance components."""
    cdef public tuple subifds # tuple[int, ...]
    """Positions of SubIFDs in file."""
    cdef public bytes jpegtables
    """JPEG quantization and Huffman tables."""
    cdef public bytes jpegheader
    """JPEG header for NDPI."""
    cdef public str software
    """Software used to create image."""
    cdef public str description
    """Subject of image."""
    cdef public str description1
    """Value of second ImageDescription tag."""
    cdef public float nodata # int | float
    """Value used for missing data. The value of the GDAL_NODATA tag or 0."""

'''