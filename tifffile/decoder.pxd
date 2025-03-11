#cython: language_level=3

from libc.stdint cimport int64_t
import numpy

from .pages cimport TiffPage

cdef class TiffDecoder:
    """Base class for TIFF segment decoders."""
    
    cdef TiffPage page
    
    #cpdef tuple __call__(self, object data, int64_t index, **kwargs)

    @staticmethod
    cdef TiffDecoder create(TiffPage page)

cdef class TiffDecoderError(TiffDecoder):
    """Decoder that raises an error."""
    cdef str error_message
    
cdef class TiffDecoderJpeg(TiffDecoder):
    """Decoder for JPEG compressed segments."""
    cdef object colorspace
    cdef object outcolorspace
    cdef object indices_func
    cdef object reshape_func
    cdef object pad_func
    cdef object pad_none_func
    
cdef class TiffDecoderEer(TiffDecoder):
    """Decoder for EER compressed segments."""
    cdef object decompress
    cdef int64_t rlebits
    cdef int64_t horzbits
    cdef int64_t vertbits
    cdef object indices_func
    cdef object pad_none_func
    
cdef class TiffDecoderJetraw(TiffDecoder):
    """Decoder for Jetraw compressed segments."""
    cdef object decompress
    cdef object indices_func
    cdef object pad_none_func
    
cdef class TiffDecoderImage(TiffDecoder):
    """Decoder for image compressions."""
    cdef object decompress
    cdef object indices_func
    cdef object reshape_func
    cdef object pad_func
    cdef object pad_none_func
    
cdef class TiffDecoderOther(TiffDecoder):
    """Decoder for other formats."""
    cdef object decompress
    cdef object unpack
    cdef object unpredict
    cdef object indices_func
    cdef object reshape_func
    cdef object pad_func
    cdef object pad_none_func
    cdef int64_t fillorder
    cdef object dtype
