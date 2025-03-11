#cython: language_level=3
#cython: boundscheck=True
#cython: wraparound=True
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t

import numpy
import imagecodecs

cimport cython
from .pages cimport TiffPage

from .format cimport ByteOrder, TiffFormat
from .files cimport FileHandle
from .tags cimport TiffTag, TiffTags
from .tags import read_uic1tag
from .types import COMPRESSION, PHOTOMETRIC, SAMPLEFORMAT, PREDICTOR,\
    EXTRASAMPLE, RESUNIT, TiffFileError, TIFF
from .utils cimport product
from .utils import logger, enumarg,\
    pformat, astype, strptime, apply_colormap,\
    create_output, stripnull, bytes2str,\
    jpeg_decode_colorspace, identityfunc

try:
    import imagecodecs
except ImportError:
    # load pure Python implementation of some codecs
    try:
        from . import _imagecodecs as imagecodecs  # type: ignore[no-redef]
    except ImportError:
        import _imagecodecs as imagecodecs  # type: ignore[no-redef]

@cython.final
cdef class CompressionCodec:
    """Map :py:class:`COMPRESSION` value to encode or decode function.

    Parameters:
        encode: If *True*, return encode functions, else decode functions.

    """

    cdef dict _codecs#: dict[int, Callable[..., Any]]
    cdef bint _encode

    def __init__(self, encode: bool) -> None:
        self._codecs = {1: identityfunc}
        self._encode = bool(encode)

    def __getitem__(self, key_obj) -> Callable[..., Any]:
        if key_obj in self._codecs:
            return self._codecs[key_obj]
        cdef int64_t key = key_obj
        codec: Callable[..., Any]
        try:
            # TODO: enable CCITTRLE decoder for future imagecodecs
            # if key == 2:
            #     if self._encode:
            #         codec = imagecodecs.ccittrle_encode
            #     else:
            #         codec = imagecodecs.ccittrle_decode
            if key == 5:
                if self._encode:
                    codec = imagecodecs.lzw_encode
                else:
                    codec = imagecodecs.lzw_decode
            elif key in {6, 7, 33007}:
                if self._encode:
                    if key in {6, 33007}:
                        raise NotImplementedError
                    codec = imagecodecs.jpeg_encode
                else:
                    codec = imagecodecs.jpeg_decode
            elif key in {8, 32946, 50013}:
                if (
                    hasattr(imagecodecs, 'DEFLATE')
                    and imagecodecs.DEFLATE.available
                ):
                    # imagecodecs built with deflate
                    if self._encode:
                        codec = imagecodecs.deflate_encode
                    else:
                        codec = imagecodecs.deflate_decode
                elif (
                    hasattr(imagecodecs, 'ZLIB') and imagecodecs.ZLIB.available
                ):
                    if self._encode:
                        codec = imagecodecs.zlib_encode
                    else:
                        codec = imagecodecs.zlib_decode
                else:
                    # imagecodecs built without zlib
                    try:
                        from . import _imagecodecs
                    except ImportError:
                        import _imagecodecs  # type: ignore[no-redef]

                    if self._encode:
                        codec = _imagecodecs.zlib_encode
                    else:
                        codec = _imagecodecs.zlib_decode
            elif key == 32773:
                if self._encode:
                    codec = imagecodecs.packbits_encode
                else:
                    codec = imagecodecs.packbits_decode
            elif key in {33003, 33004, 33005, 34712}:
                if self._encode:
                    codec = imagecodecs.jpeg2k_encode
                else:
                    codec = imagecodecs.jpeg2k_decode
            elif key == 34887:
                if self._encode:
                    codec = imagecodecs.lerc_encode
                else:
                    codec = imagecodecs.lerc_decode
            elif key == 34892:
                # DNG lossy
                if self._encode:
                    codec = imagecodecs.jpeg8_encode
                else:
                    codec = imagecodecs.jpeg8_decode
            elif key == 34925:
                if hasattr(imagecodecs, 'LZMA') and imagecodecs.LZMA.available:
                    if self._encode:
                        codec = imagecodecs.lzma_encode
                    else:
                        codec = imagecodecs.lzma_decode
                else:
                    # imagecodecs built without lzma
                    try:
                        from . import _imagecodecs
                    except ImportError:
                        import _imagecodecs  # type: ignore[no-redef]

                    if self._encode:
                        codec = _imagecodecs.lzma_encode
                    else:
                        codec = _imagecodecs.lzma_decode
            elif key == 34933:
                if self._encode:
                    codec = imagecodecs.png_encode
                else:
                    codec = imagecodecs.png_decode
            elif key in {34934, 22610}:
                if self._encode:
                    codec = imagecodecs.jpegxr_encode
                else:
                    codec = imagecodecs.jpegxr_decode
            elif key == 48124:
                if self._encode:
                    codec = imagecodecs.jetraw_encode
                else:
                    codec = imagecodecs.jetraw_decode
            elif key in {50000, 34926}:  # 34926 deprecated
                if self._encode:
                    codec = imagecodecs.zstd_encode
                else:
                    codec = imagecodecs.zstd_decode
            elif key in {50001, 34927}:  # 34927 deprecated
                if self._encode:
                    codec = imagecodecs.webp_encode
                else:
                    codec = imagecodecs.webp_decode
            elif key in {65000, 65001, 65002} and not self._encode:
                codec = imagecodecs.eer_decode
            elif key in {50002, 52546}:
                if self._encode:
                    codec = imagecodecs.jpegxl_encode
                else:
                    codec = imagecodecs.jpegxl_decode
            else:
                try:
                    msg = f'{COMPRESSION(key)!r} not supported'
                except ValueError:
                    msg = f'{key} is not a known COMPRESSION'
                raise KeyError(msg)
        except (AttributeError, ImportError) as exc:
            raise KeyError(
                f'{COMPRESSION(key)!r} ' "requires the 'imagecodecs' package"
            ) from exc
        except NotImplementedError as exc:
            raise KeyError(f'{COMPRESSION(key)!r} not implemented') from exc
        self._codecs[key] = codec
        return codec

    def __contains__(self, key: Any) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[int]:
        yield 1  # dummy

    def __len__(self) -> int:
        return 1  # dummy


@cython.final
cdef class PredictorCodec:
    """Map :py:class:`PREDICTOR` value to encode or decode function.

    Parameters:
        encode: If *True*, return encode functions, else decode functions.

    """

    cdef dict _codecs#: dict[int, Callable[..., Any]]
    cdef bint _encode

    def __init__(self, encode: bool) -> None:
        self._codecs = {1: identityfunc}
        self._encode = bool(encode)

    def __getitem__(self, key: int) -> Callable[..., Any]:
        if key in self._codecs:
            return self._codecs[key]
        codec: Callable[..., Any]
        try:
            if key == 2:
                if self._encode:
                    codec = imagecodecs.delta_encode
                else:
                    codec = imagecodecs.delta_decode
            elif key == 3:
                if self._encode:
                    codec = imagecodecs.floatpred_encode
                else:
                    codec = imagecodecs.floatpred_decode
            elif key == 34892:
                if self._encode:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.delta_encode(
                            data, axis=axis, out=out, dist=2
                        )

                else:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.delta_decode(
                            data, axis=axis, out=out, dist=2
                        )

            elif key == 34893:
                if self._encode:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.delta_encode(
                            data, axis=axis, out=out, dist=4
                        )

                else:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.delta_decode(
                            data, axis=axis, out=out, dist=4
                        )

            elif key == 34894:
                if self._encode:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.floatpred_encode(
                            data, axis=axis, out=out, dist=2
                        )

                else:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.floatpred_decode(
                            data, axis=axis, out=out, dist=2
                        )

            elif key == 34895:
                if self._encode:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.floatpred_encode(
                            data, axis=axis, out=out, dist=4
                        )

                else:

                    def codec(data, axis=-1, out=None):
                        return imagecodecs.floatpred_decode(
                            data, axis=axis, out=out, dist=4
                        )

            else:
                raise KeyError(f'{key} is not a known PREDICTOR')
        except AttributeError as exc:
            raise KeyError(
                f'{PREDICTOR(key)!r}' " requires the 'imagecodecs' package"
            ) from exc
        except NotImplementedError as exc:
            raise KeyError(f'{PREDICTOR(key)!r} not implemented') from exc
        self._codecs[key] = codec
        return codec

    def __contains__(self, key: Any) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[int]:
        yield 1  # dummy

    def __len__(self) -> int:
        return 1  # dummy

PREDICTORS = PredictorCodec(True)
UNPREDICTORS = PredictorCodec(False)
COMPRESSORS = CompressionCodec(True)
DECOMPRESSORS = CompressionCodec(False)


def unpack_rgb(
    data: bytes,
    dtype: DTypeLike | None = None,
    bitspersample: tuple[int, ...] | None = None,
    rescale: bool = True,
) -> NDArray[Any]:
    """Return array from bytes containing packed samples.

    Use to unpack RGB565 or RGB555 to RGB888 format.
    Works on little-endian platforms only.

    Parameters:
        data:
            Bytes to be decoded.
            Samples in each pixel are stored consecutively.
            Pixels are aligned to 8, 16, or 32 bit boundaries.
        dtype:
            Data type of samples.
            The byte order applies also to the data stream.
        bitspersample:
            Number of bits for each sample in pixel.
        rescale:
            Upscale samples to number of bits in dtype.

    Returns:
        Flattened array of unpacked samples of native dtype.

    Examples:
        >>> data = struct.pack('BBBB', 0x21, 0x08, 0xFF, 0xFF)
        >>> print(unpack_rgb(data, '<B', (5, 6, 5), False))
        [ 1  1  1 31 63 31]
        >>> print(unpack_rgb(data, '<B', (5, 6, 5)))
        [  8   4   8 255 255 255]
        >>> print(unpack_rgb(data, '<B', (5, 5, 5)))
        [ 16   8   8 255 255 255]

    """
    cdef int64_t bits, i, bps
    cdef int64_t o
    cdef str dt
    cdef object data_array, result, t
    
    if bitspersample is None:
        bitspersample = (5, 6, 5)
    if dtype is None:
        dtype = '<B'
    dtype = numpy.dtype(dtype)
    bits = int(numpy.sum(bitspersample))
    if not (
        bits <= 32 and all(i <= dtype.itemsize * 8 for i in bitspersample)
    ):
        raise ValueError(f'sample size not supported: {bitspersample}')
    dt = next(i for i in 'BHI' if numpy.dtype(i).itemsize * 8 >= bits)
    data_array = numpy.frombuffer(data, dtype.byteorder + dt)
    result = numpy.empty((data_array.size, len(bitspersample)), dtype.char)
    for i, bps in enumerate(bitspersample):
        t = data_array >> int(numpy.sum(bitspersample[i + 1 :]))
        t &= int('0b' + '1' * bps, 2)
        if rescale:
            o = ((dtype.itemsize * 8) // bps + 1) * bps
            if o > data_array.dtype.itemsize * 8:
                t = t.astype('I')
            t *= (2**o - 1) // (2**bps - 1)
            t //= 2 ** (o - (dtype.itemsize * 8))
        result[:, i] = t
    return result.reshape(-1)

cdef class TiffDecoder:
    """Base class for TIFF segment decoders."""
    
    def __init__(self, TiffPage page):
        self.page = page
        
    def __call__(self, object data, int64_t index, **kwargs):
        """Decode segment data.
        
        Parameters:
            data: Encoded bytes of segment or None for empty segments.
            index: Index of segment in Offsets and Bytecount tag values.
            **kwargs: Additional arguments for specific decoders.
        
        Returns:
            tuple: (decoded_data, segment_indices, segment_shape)
        """
        raise NotImplementedError("Subclass must implement __call__")

    @staticmethod
    cdef TiffDecoder create(TiffPage page):
        """Create appropriate decoder for the TIFF page.
        
        Parameters:
            page: TiffPage instance to create decoder for.
            
        Returns:
            TiffDecoder: Instance configured for the page.
            
        Raises:
            ValueError: If decoding is not supported.
        """
        cdef int64_t stdepth, stlength, stwidth, samples, imdepth, imlength, imwidth
        cdef int64_t width, length, depth
        
        # Check common error conditions first
        if page.dtype is None or page._dtype is None:
            return TiffDecoderError(
                page, 
                'data type not supported '
                f'(SampleFormat {page.sampleformat}, '
                f'{page.bitspersample}-bit)'
            )

        if 0 in tuple(page.shaped):
            return TiffDecoderError(page, 'empty image')

        # Get decompressor function
        try:
            if page.compression == 1:
                decompress = None
            else:
                decompress = DECOMPRESSORS[page.compression]
            if (
                page.compression in {65000, 65001, 65002}
                and not page.parent.is_eer
            ):
                raise KeyError(page.compression)
        except KeyError as exc:
            return TiffDecoderError(page, str(exc)[1:-1])

        # Get unpredictor function
        try:
            if page.predictor == 1:
                unpredict = None
            else:
                unpredict = UNPREDICTORS[page.predictor]
        except KeyError as exc:
            if page.compression in TIFF.IMAGE_COMPRESSIONS:
                logger().warning(
                    f'{page!r} ignoring predictor {page.predictor}'
                )
                unpredict = None
            else:
                return TiffDecoderError(page, str(exc)[1:-1])

        # Check if sample formats match
        if page.tags.get(339) is not None:
            tag = page.tags[339]  # SampleFormat
            if tag.count != 1 and any(i - tag.value[0] for i in tag.value_get()):
                return TiffDecoderError(
                    page, f'sample formats do not match {tag.value}'
                )

        # Check chroma subsampling
        if page.is_subsampled() and (
            page.compression not in {6, 7, 34892, 33007}
            or page.planarconfig == 2
        ):
            return TiffDecoderError(
                page, 'chroma subsampling not supported without JPEG compression'
            )

        # Special case for WebP with alpha channel
        if page.compression == 50001 and page.samplesperpixel == 4:
            # WebP segments may be missing all-opaque alpha channel
            def decompress_webp_rgba(data, out=None):
                return imagecodecs.webp_decode(data, hasalpha=True, out=out)
            decompress = decompress_webp_rgba

        # Normalize segments shape to [depth, length, width, contig]
        if page.is_tiled():
            stshape = (
                page.tiledepth,
                page.tilelength,
                page.tilewidth,
                page.samplesperpixel if page.planarconfig == 1 else 1,
            )
        else:
            stshape = (
                1,
                page.rowsperstrip,
                page.imagewidth,
                page.samplesperpixel if page.planarconfig == 1 else 1,
            )

        stdepth, stlength, stwidth, samples = stshape
        _, imdepth, imlength, imwidth, _ = page.shaped

        # Create helper functions based on tiled or strip mode
        if page.is_tiled():
            width = (imwidth + stwidth - 1) // stwidth
            length = (imlength + stlength - 1) // stlength
            depth = (imdepth + stdepth - 1) // stdepth
            
            indices_func = create_tile_indices_func(
                width, length, depth, stdepth, stlength, stwidth, samples
            )
            reshape_func = create_reshape_tile_func(
                imdepth, imlength, imwidth, samples
            )
            pad_func = create_pad_tile_func(page.nodata)
            pad_none_func = create_pad_none_tile_func()
        else:
            # strips
            length = (imlength + stlength - 1) // stlength
            
            indices_func = create_strip_indices_func(
                length, imdepth, stdepth, stlength, stwidth, samples
            )
            reshape_func = create_reshape_strip_func()
            pad_func = create_pad_strip_func(stlength, page.nodata)
            pad_none_func = create_pad_none_strip_func(stlength)

        # Select appropriate decoder based on compression
        if page.compression in {6, 7, 34892, 33007}:
            # JPEG needs special handling
            if page.fillorder == 2:
                logger().debug(f'{page!r} disabling LSB2MSB for JPEG')
            if unpredict:
                logger().debug(f'{page!r} disabling predictor for JPEG')
            if 28672 in page.tags:  # SonyRawFileType
                logger().warning(
                    f'{page!r} SonyRawFileType might need additional '
                    'unpacking (see issue #95)'
                )
                
            return TiffDecoderJpeg(
                page, indices_func, reshape_func, pad_func, pad_none_func
            )
                
        elif page.compression in {65000, 65001, 65002}:
            # EER decoder requires shape and extra args
            if page.compression == 65002:
                rlebits = int(page.tags.valueof(65007, 7))
                horzbits = int(page.tags.valueof(65008, 2))
                vertbits = int(page.tags.valueof(65009, 2))
            elif page.compression == 65001:
                rlebits = 7
                horzbits = 2
                vertbits = 2
            else:
                rlebits = 8
                horzbits = 2
                vertbits = 2
                
            return TiffDecoderEer(
                page, decompress, rlebits, horzbits, vertbits, 
                indices_func, pad_none_func
            )
                
        elif page.compression == 48124:
            # Jetraw requires pre-allocated output buffer
            return TiffDecoderJetraw(
                page, decompress, indices_func, pad_none_func
            )
                
        elif page.compression in TIFF.IMAGE_COMPRESSIONS:
            # presume codecs always return correct dtype, native byte order...
            if page.fillorder == 2:
                logger().debug(
                    f'{page!r} '
                    f'disabling LSB2MSB for compression {page.compression}'
                )
            if unpredict:
                logger().debug(
                    f'{page!r} '
                    f'disabling predictor for compression {page.compression}'
                )
                
            return TiffDecoderImage(
                page, decompress, indices_func, reshape_func, pad_func, pad_none_func
            )
                
        else:
            # Regular data types
            dtype = numpy.dtype(page.parent.byteorder + page._dtype.char)

            if page.sampleformat == 5:
                # complex integer
                if unpredict is not None:
                    raise NotImplementedError(
                        'unpredicting complex integers not supported'
                    )

                itype = numpy.dtype(
                    f'{page.parent.byteorder}i{page.bitspersample // 16}'
                )
                ftype = numpy.dtype(
                    f'{page.parent.byteorder}f{dtype.itemsize // 2}'
                )

                def unpack(data):
                    # return complex integer as numpy.complex
                    return numpy.frombuffer(data, itype).astype(ftype).view(dtype)

            elif page.bitspersample in {8, 16, 32, 64, 128}:
                # regular data types
                if (page.bitspersample * stwidth * samples) % 8:
                    raise ValueError('data and sample size mismatch')
                if page.predictor in {3, 34894, 34895}:  # PREDICTOR.FLOATINGPOINT
                    # floating-point horizontal differencing decoder needs
                    # raw byte order
                    dtype = numpy.dtype(page._dtype.char)

                def unpack(data):
                    # return numpy array from buffer
                    try:
                        # read only numpy array
                        return numpy.frombuffer(data, dtype)
                    except ValueError:
                        # for example, LZW strips may be missing EOI
                        bps = page.bitspersample // 8
                        size = (len(data) // bps) * bps
                        return numpy.frombuffer(data[:size], dtype)

            elif isinstance(page.bitspersample, tuple):
                # for example, RGB 565
                def unpack(data):
                    # return numpy array from packed integers
                    return unpack_rgb(data, dtype, page.bitspersample)

            elif page.bitspersample == 24 and dtype.char == 'f':
                # float24
                if unpredict is not None:
                    # floatpred_decode requires numpy.float24, which does not exist
                    raise NotImplementedError('unpredicting float24 not supported')

                def unpack(data):
                    # return numpy.float32 array from float24
                    return imagecodecs.float24_decode(
                        data, byteorder=page.parent.byteorder
                    )

            else:
                # bilevel and packed integers
                def unpack(data):
                    # return NumPy array from packed integers
                    return imagecodecs.packints_decode(
                        data, dtype, page.bitspersample, runlen=stwidth * samples
                    )
                    
            return TiffDecoderOther(
                page, decompress, unpack, unpredict, page.fillorder, dtype,
                indices_func, reshape_func, pad_func, pad_none_func
            )

cdef class TiffDecoderError(TiffDecoder):
    """Decoder that raises an error."""
    
    def __init__(self, TiffPage page, str error_message):
        super().__init__(page)
        self.error_message = error_message
        
    def __call__(self, object data, int64_t index, **kwargs):
        raise ValueError(self.error_message)

cdef class TiffDecoderJpeg(TiffDecoder):
    """Decoder for JPEG compressed segments."""
    
    def __init__(self, TiffPage page, object indices_func, object reshape_func, 
                 object pad_func, object pad_none_func):
        super().__init__(page)
        self.colorspace, self.outcolorspace = jpeg_decode_colorspace(
            page.photometric, page.planarconfig, page.extrasamples, page.is_jfif()
        )
        self.indices_func = indices_func
        self.reshape_func = reshape_func
        self.pad_func = pad_func
        self.pad_none_func = pad_none_func
        
    def __call__(self, object data, int64_t index, **kwargs):
        cdef bint _fullsize = kwargs.get('_fullsize', False)
        cdef object jpegtables = kwargs.get('jpegtables', None)
        cdef object jpegheader = kwargs.get('jpegheader', None)
        
        segmentindex, shape = self.indices_func(index)
        if data is None:
            if _fullsize:
                shape = self.pad_none_func(shape)
            return data, segmentindex, shape
            
        data_array = imagecodecs.jpeg_decode(
            data,
            bitspersample=self.page.bitspersample,
            tables=jpegtables,
            header=jpegheader,
            colorspace=self.colorspace,
            outcolorspace=self.outcolorspace,
            shape=shape[1:3]
        )
        data_array = self.reshape_func(data_array, segmentindex, shape)
        if _fullsize:
            data_array, shape = self.pad_func(data_array, shape)
        return data_array, segmentindex, shape

cdef class TiffDecoderEer(TiffDecoder):
    """Decoder for EER compressed segments."""
    
    def __init__(self, TiffPage page, object decompress, int64_t rlebits, 
                 int64_t horzbits, int64_t vertbits, object indices_func, 
                 object pad_none_func):
        super().__init__(page)
        self.decompress = decompress
        self.rlebits = rlebits
        self.horzbits = horzbits
        self.vertbits = vertbits
        self.indices_func = indices_func
        self.pad_none_func = pad_none_func
        
    def __call__(self, object data, int64_t index, **kwargs):
        cdef bint _fullsize = kwargs.get('_fullsize', False)
        
        segmentindex, shape = self.indices_func(index)
        if data is None:
            if _fullsize:
                shape = self.pad_none_func(shape)
            return data, segmentindex, shape
            
        data_array = self.decompress(
            data,
            shape=shape[1:3],
            rlebits=self.rlebits,
            horzbits=self.horzbits,
            vertbits=self.vertbits,
            superres=False
        )
        return data_array.reshape(shape), segmentindex, shape

cdef class TiffDecoderJetraw(TiffDecoder):
    """Decoder for Jetraw compressed segments."""
    
    def __init__(self, TiffPage page, object decompress, object indices_func, 
                 object pad_none_func):
        super().__init__(page)
        self.decompress = decompress
        self.indices_func = indices_func
        self.pad_none_func = pad_none_func
        
    def __call__(self, object data, int64_t index, **kwargs):
        cdef bint _fullsize = kwargs.get('_fullsize', False)
        
        segmentindex, shape = self.indices_func(index)
        if data is None:
            if _fullsize:
                shape = self.pad_none_func(shape)
            return data, segmentindex, shape
            
        data_array = numpy.zeros(shape, numpy.uint16)
        self.decompress(data, out=data_array)
        return data_array.reshape(shape), segmentindex, shape

cdef class TiffDecoderImage(TiffDecoder):
    """Decoder for image compressions."""
    
    def __init__(self, TiffPage page, object decompress, object indices_func, 
                 object reshape_func, object pad_func, object pad_none_func):
        super().__init__(page)
        self.decompress = decompress
        self.indices_func = indices_func
        self.reshape_func = reshape_func
        self.pad_func = pad_func
        self.pad_none_func = pad_none_func
        
    def __call__(self, object data, int64_t index, **kwargs):
        cdef bint _fullsize = kwargs.get('_fullsize', False)
        
        segmentindex, shape = self.indices_func(index)
        if data is None:
            if _fullsize:
                shape = self.pad_none_func(shape)
            return data, segmentindex, shape
            
        data_array = self.decompress(data)
        data_array = self.reshape_func(data_array, segmentindex, shape)
        if _fullsize:
            data_array, shape = self.pad_func(data_array, shape)
        return data_array, segmentindex, shape

cdef class TiffDecoderOther(TiffDecoder):
    """Decoder for other formats."""
    
    def __init__(self, TiffPage page, object decompress, object unpack, 
                 object unpredict, int64_t fillorder, object dtype,
                 object indices_func, object reshape_func, object pad_func, 
                 object pad_none_func):
        super().__init__(page)
        self.decompress = decompress
        self.unpack = unpack
        self.unpredict = unpredict
        self.fillorder = fillorder
        self.dtype = dtype
        self.indices_func = indices_func
        self.reshape_func = reshape_func
        self.pad_func = pad_func
        self.pad_none_func = pad_none_func
        
    def __call__(self, object data, int64_t index, **kwargs):
        cdef bint _fullsize = kwargs.get('_fullsize', False)
        
        segmentindex, shape = self.indices_func(index)
        if data is None:
            if _fullsize:
                shape = self.pad_none_func(shape)
            return data, segmentindex, shape
            
        if self.fillorder == 2:
            data = imagecodecs.bitorder_decode(data)
        if self.decompress is not None:
            size = shape[0] * shape[1] * shape[2] * shape[3]
            data = self.decompress(data, out=size * self.dtype.itemsize)
            
        data_array = self.unpack(data)
        data_array = self.reshape_func(data_array, segmentindex, shape)
        data_array = data_array.astype('=' + self.dtype.char, copy=False)
        if self.unpredict is not None:
            data_array = self.unpredict(data_array, axis=-2, out=data_array)
        if _fullsize:
            data_array, shape = self.pad_func(data_array, shape)
        return data_array, segmentindex, shape

# Helper functions for creating index and shape handling functions
def create_tile_indices_func(width, length, depth, stdepth, stlength, stwidth, samples):
    """Create function to return indices and shape of tile."""
    def indices_func(segmentindex):
        return (
            (
                segmentindex // (width * length * depth),
                (segmentindex // (width * length)) % depth * stdepth,
                (segmentindex // width) % length * stlength,
                segmentindex % width * stwidth,
                0,
            ),
            (stdepth, stlength, stwidth, samples),
        )
    return indices_func

def create_strip_indices_func(length, imdepth, stdepth, stlength, stwidth, samples):
    """Create function to return indices and shape of strip."""
    def indices_func(segmentindex):
        indices = (
            segmentindex // (length * imdepth),
            (segmentindex // length) % imdepth * stdepth,
            segmentindex % length * stlength,
            0,
            0,
        )
        shape = (
            stdepth,
            min(stlength, stlength - indices[2] % stlength),
            stwidth,
            samples,
        )
        return indices, shape
    return indices_func

def create_reshape_tile_func(imdepth, imlength, imwidth, samples):
    """Create function to reshape tile data."""
    def reshape_func(data, indices, shape):
        size = shape[0] * shape[1] * shape[2] * shape[3]
        if data.ndim == 1 and data.size > size:
            data = data[:size]
        if data.size == size:
            return data.reshape(shape)
        try:
            return data.reshape(
                (
                    min(imdepth - indices[1], shape[0]),
                    min(imlength - indices[2], shape[1]),
                    min(imwidth - indices[3], shape[2]),
                    samples,
                )
            )
        except ValueError:
            pass
        try:
            return data.reshape(
                (
                    min(imdepth - indices[1], shape[0]),
                    min(imlength - indices[2], shape[1]),
                    shape[2],
                    samples,
                )
            )
        except ValueError:
            pass
        raise TiffFileError(
            f'corrupted tile @ {indices} cannot be reshaped from '
            f'{data.shape} to {shape}'
        )
    return reshape_func

def create_reshape_strip_func():
    """Create function to reshape strip data."""
    def reshape_func(data, indices, shape):
        size = shape[0] * shape[1] * shape[2] * shape[3]
        if data.ndim == 1 and data.size > size:
            data = data[:size]
        if data.size == size:
            try:
                data.shape = shape
            except AttributeError:
                data = data.reshape(shape)
            return data
        datashape = data.shape
        try:
            data.shape = shape[0], -1, shape[2], shape[3]
            data = data[:, : shape[1]]
            data.shape = shape
            return data
        except ValueError:
            pass
        raise TiffFileError(
            'corrupted strip cannot be reshaped from '
            f'{datashape} to {shape}'
        )
    return reshape_func

def create_pad_tile_func(nodata):
    """Create function to pad tile to shape."""
    def pad_func(data, shape):
        if data.shape == shape:
            return data, shape
        padwidth = [(0, i - j) for i, j in zip(shape, data.shape)]
        data = numpy.pad(data, padwidth, constant_values=nodata)
        return data, shape
    return pad_func

def create_pad_none_tile_func():
    """Create function to return shape of tile."""
    def pad_none_func(shape):
        return shape
    return pad_none_func

def create_pad_strip_func(stlength, nodata):
    """Create function to pad strip to shape."""
    def pad_func(data, shape):
        shape = (shape[0], stlength, shape[2], shape[3])
        if data.shape == shape:
            return data, shape
        padwidth = [
            (0, 0),
            (0, stlength - data.shape[1]),
            (0, 0),
            (0, 0),
        ]
        data = numpy.pad(data, padwidth, constant_values=nodata)
        return data, shape
    return pad_func

def create_pad_none_strip_func(stlength):
    """Create function to return shape of strip."""
    def pad_none_func(shape):
        return (shape[0], stlength, shape[2], shape[3])
    return pad_none_func
