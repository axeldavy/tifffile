#cython: language_level=3

from libc.stdint cimport int64_t
import numpy

from .pages cimport TiffPage

cdef class TiffDecoder:
    """Base class for TIFF segment decoders."""
    
    cdef TiffPage page
    # Common shape and dimension attributes needed for all decoders
    cdef int64_t width
    cdef int64_t length
    cdef int64_t depth
    cdef int64_t imdepth
    cdef int64_t imlength
    cdef int64_t imwidth
    cdef int64_t stdepth
    cdef int64_t stlength
    cdef int64_t stwidth
    cdef int64_t samples
    cdef object nodata
    cdef bint is_tiled
    
    cdef tuple get_indices_shape(self, int64_t segmentindex)
    cdef object reshape_data(self, object data, tuple indices, tuple shape)
    cdef tuple pad_data(self, object data, tuple shape)
    cdef tuple pad_none(self, tuple shape)
    
    @staticmethod
    cdef TiffDecoder create(TiffPage page)

cdef class TiffDecoderError(TiffDecoder):
    """Decoder that raises an error."""
    cdef str error_message
    @staticmethod
    cdef TiffDecoderError initialize(TiffPage page, str error_message)
    
cdef class TiffDecoderJpeg(TiffDecoder):
    """Decoder for JPEG compressed segments."""
    cdef object colorspace
    cdef object outcolorspace
    @staticmethod
    cdef TiffDecoderJpeg initialize(TiffPage page, object colorspace, object outcolorspace)
    
cdef class TiffDecoderEer(TiffDecoder):
    """Decoder for EER compressed segments."""
    cdef object decompress
    cdef int64_t rlebits
    cdef int64_t horzbits
    cdef int64_t vertbits
    @staticmethod
    cdef TiffDecoderEer initialize(TiffPage page, object decompress, int64_t rlebits, int64_t horzbits, int64_t vertbits)
    
cdef class TiffDecoderJetraw(TiffDecoder):
    """Decoder for Jetraw compressed segments."""
    cdef object decompress
    @staticmethod
    cdef TiffDecoderJetraw initialize(TiffPage page, object decompress)
    
cdef class TiffDecoderImage(TiffDecoder):
    """Decoder for image compressions."""
    cdef object decompress
    @staticmethod
    cdef TiffDecoderImage initialize(TiffPage page, object decompress)
    
cdef class TiffDecoderBase(TiffDecoder):
    """Base class for other format decoders."""
    cdef object decompress
    cdef object unpredict
    cdef int64_t fillorder
    cdef object dtype
    cdef object unpack(self, object data)

cdef class TiffDecoderComplexInt(TiffDecoderBase):
    """Decoder for complex integers."""
    cdef object itype
    cdef object ftype
    cdef object unpack(self, object data)
    @staticmethod
    cdef TiffDecoderComplexInt initialize(TiffPage page, object decompress, object unpredict, int64_t fillorder, object dtype, object itype, object ftype)
    
cdef class TiffDecoderRegular(TiffDecoderBase):
    """Decoder for regular data types."""
    cdef object unpack(self, object data)
    @staticmethod
    cdef TiffDecoderRegular initialize(TiffPage page, object decompress, object unpredict, int64_t fillorder, object dtype)
    
cdef class TiffDecoderRGB(TiffDecoderBase):
    """Decoder for RGB packed integers."""
    cdef tuple bitspersample_rgb
    cdef object unpack(self, object data)
    @staticmethod
    cdef TiffDecoderRGB initialize(TiffPage page, object decompress, object unpredict, int64_t fillorder, object dtype, tuple bitspersample_rgb)
    
cdef class TiffDecoderFloat24(TiffDecoderBase):
    """Decoder for float24 data type."""
    cdef object unpack(self, object data)
    @staticmethod
    cdef TiffDecoderFloat24 initialize(TiffPage page, object decompress, object unpredict, int64_t fillorder, object dtype)
    
cdef class TiffDecoderPackedBits(TiffDecoderBase):
    """Decoder for bilevel and packed integers."""
    cdef int64_t bitspersample
    cdef int64_t runlen
    cdef object unpack(self, object data)
    @staticmethod
    cdef TiffDecoderPackedBits initialize(TiffPage page, object decompress, object unpredict, int64_t fillorder, object dtype, int64_t bitspersample, int64_t runlen)
