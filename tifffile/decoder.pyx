#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=True
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from libcpp.vector cimport vector

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
        # Initialize shape parameters
        if page.is_tiled():
            self.is_tiled = True
            self.stdepth = page.tiledepth
            self.stlength = page.tilelength
            self.stwidth = page.tilewidth
            _, self.imdepth, self.imlength, self.imwidth, _ = page.shaped
            self.samples = page.samplesperpixel if page.planarconfig == 1 else 1
            self.width = (self.imwidth + self.stwidth - 1) // self.stwidth
            self.length = (self.imlength + self.stlength - 1) // self.stlength
            self.depth = (self.imdepth + self.stdepth - 1) // self.stdepth
        else:
            self.is_tiled = False
            self.stdepth = 1
            self.stlength = page.rowsperstrip
            self.stwidth = page.imagewidth
            _, self.imdepth, self.imlength, self.imwidth, _ = page.shaped
            self.samples = page.samplesperpixel if page.planarconfig == 1 else 1
            self.length = (self.imlength + self.stlength - 1) // self.stlength
            self.width = 1
            self.depth = 1
        self.nodata = page.nodata
        
    def __call__(self, FileHandle fh, int64_t offset, int64_t read_len, int64_t index, **kwargs):
        """Decode segment data.
        
        Parameters:
            data: Encoded bytes of segment or None for empty segments.
            index: Index of segment in Offsets and Bytecount tag values.
            **kwargs: Additional arguments for specific decoders.
        
        Returns:
            tuple: (decoded_data, segment_indices, segment_shape)
        """
        raise NotImplementedError("Subclass must implement __call__")
    
    cdef tuple get_indices_shape(self, int64_t segmentindex):
        """Get indices and shape for a segment."""
        cdef tuple indices
        cdef tuple shape
        
        if self.is_tiled:
            # Tile indices
            indices = (
                segmentindex // (self.width * self.length * self.depth),
                (segmentindex // (self.width * self.length)) % self.depth * self.stdepth,
                (segmentindex // self.width) % self.length * self.stlength,
                segmentindex % self.width * self.stwidth,
                0,
            )
            shape = (self.stdepth, self.stlength, self.stwidth, self.samples)
        else:
            # Strip indices
            indices = (
                segmentindex // (self.length * self.imdepth),
                (segmentindex // self.length) % self.imdepth * self.stdepth,
                segmentindex % self.length * self.stlength,
                0,
                0,
            )
            shape = (
                self.stdepth,
                min(self.stlength, self.stlength - indices[2] % self.stlength),
                self.stwidth,
                self.samples,
            )
        return indices, shape
    
    cdef object reshape_data(self, object data, tuple indices, tuple shape):
        """Reshape data array according to indices and shape."""
        cdef int64_t size = shape[0] * shape[1] * shape[2] * shape[3]
        
        if self.is_tiled:
            # Reshape tile data
            if data.ndim == 1 and data.size > size:
                data = data[:size]
            if data.size == size:
                return data.reshape(shape)
            try:
                return data.reshape(
                    (
                        min(self.imdepth - indices[1], shape[0]),
                        min(self.imlength - indices[2], shape[1]),
                        min(self.imwidth - indices[3], shape[2]),
                        self.samples,
                    )
                )
            except ValueError:
                pass
            try:
                return data.reshape(
                    (
                        min(self.imdepth - indices[1], shape[0]),
                        min(self.imlength - indices[2], shape[1]),
                        shape[2],
                        self.samples,
                    )
                )
            except ValueError:
                pass
            raise TiffFileError(
                f'corrupted tile @ {indices} cannot be reshaped from '
                f'{data.shape} to {shape}'
            )
        else:
            # Reshape strip data
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
    
    cdef tuple pad_data(self, object data, tuple shape):
        """Pad data to match the expected shape."""
        cdef list padwidth
        
        if self.is_tiled:
            # Pad tile data
            if data.shape == shape:
                return data, shape
            padwidth = [(0, i - j) for i, j in zip(shape, data.shape)]
            data = numpy.pad(data, padwidth, constant_values=self.nodata)
            return data, shape
        else:
            # Pad strip data
            shape = (shape[0], self.stlength, shape[2], shape[3])
            if data.shape == shape:
                return data, shape
            padwidth = [
                (0, 0),
                (0, self.stlength - data.shape[1]),
                (0, 0),
                (0, 0),
            ]
            data = numpy.pad(data, padwidth, constant_values=self.nodata)
            return data, shape
    
    cdef tuple pad_none(self, tuple shape):
        """Return the full shape for empty segments."""
        if self.is_tiled:
            return shape
        else:
            return (shape[0], self.stlength, shape[2], shape[3])

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
        # Check common error conditions first
        if page.dtype is None or page._dtype is None:
            return TiffDecoderError.initialize(
                page, 
                'data type not supported '
                f'(SampleFormat {page.sampleformat}, '
                f'{page.bitspersample}-bit)'
            )

        if 0 in tuple(page.shaped):
            return TiffDecoderError.initialize(page, 'empty image')

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
            return TiffDecoderError.initialize(page, str(exc)[1:-1])

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
                return TiffDecoderError.initialize(page, str(exc)[1:-1])

        # Check if sample formats match
        if page.tags.get(339) is not None:
            tag = page.tags.get(339)  # SampleFormat
            if tag.count != 1 and any(i - tag.value[0] for i in tag.value_get()):
                return TiffDecoderError.initialize(
                    page, f'sample formats do not match {tag.value}'
                )

        # Check chroma subsampling
        if page.is_subsampled() and (
            page.compression not in {6, 7, 34892, 33007}
            or page.planarconfig == 2
        ):
            return TiffDecoderError.initialize(
                page, 'chroma subsampling not supported without JPEG compression'
            )

        # Special case for WebP with alpha channel
        if page.compression == 50001 and page.samplesperpixel == 4:
            # WebP segments may be missing all-opaque alpha channel
            def decompress_webp_rgba(data, out=None):
                return imagecodecs.webp_decode(data, hasalpha=True, out=out)
            decompress = decompress_webp_rgba

        # Select appropriate decoder based on compression
        if page.compression in {6, 7, 34892, 33007}:
            # JPEG needs special handling
            if page.fillorder == 2:
                logger().debug(f'{page!r} disabling LSB2MSB for JPEG')
            if unpredict:
                logger().debug(f'{page!r} disabling predictor for JPEG')
            if page.tags.contains_code(28672):  # SonyRawFileType
                logger().warning(
                    f'{page!r} SonyRawFileType might need additional '
                    'unpacking (see issue #95)'
                )
            
            colorspace, outcolorspace = jpeg_decode_colorspace(
                page.photometric, page.planarconfig, page.extrasamples, page.is_jfif()
            )
            
            return TiffDecoderJpeg.initialize(page, colorspace, outcolorspace)
                
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
                
            return TiffDecoderEer.initialize(page, decompress, rlebits, horzbits, vertbits)
                
        elif page.compression == 48124:
            # Jetraw requires pre-allocated output buffer
            return TiffDecoderJetraw.initialize(page, decompress)
                
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
                
            return TiffDecoderImage.initialize(page, decompress)
                
        else:
            # Regular data types
            dtype = numpy.dtype(page.tiff.byteorder_str + page._dtype.char)
            decoder = None
            
            if page.sampleformat == 5:
                # complex integer
                if unpredict is not None:
                    raise NotImplementedError(
                        'unpredicting complex integers not supported'
                    )

                itype = numpy.dtype(
                    f'{page.tiff.byteorder_str}i{page.bitspersample // 16}'
                )
                ftype = numpy.dtype(
                    f'{page.tiff.byteorder_str}f{dtype.itemsize // 2}'
                )
                
                return TiffDecoderComplexInt.initialize(page, decompress, unpredict, page.fillorder, dtype, itype, ftype)
                
            elif page.bitspersample in {8, 16, 32, 64, 128}:
                # regular data types
                if (page.bitspersample * page.imagewidth * page.samplesperpixel) % 8:
                    raise ValueError('data and sample size mismatch')
                if page.predictor in {3, 34894, 34895}:  # PREDICTOR.FLOATINGPOINT
                    # floating-point horizontal differencing decoder needs
                    # raw byte order
                    dtype = numpy.dtype(page._dtype.char)
                    
                return TiffDecoderRegular.initialize(page, decompress, unpredict, page.fillorder, dtype)

            elif isinstance(page.bitspersample, tuple):
                # for example, RGB 565
                return TiffDecoderRGB.initialize(page, decompress, unpredict, page.fillorder, dtype, page.bitspersample)

            elif page.bitspersample == 24 and dtype.char == 'f':
                # float24
                if unpredict is not None:
                    # floatpred_decode requires numpy.float24, which does not exist
                    raise NotImplementedError('unpredicting float24 not supported')
                    
                return TiffDecoderFloat24.initialize(page, decompress, unpredict, page.fillorder, dtype)

            else:
                # bilevel and packed integers
                return TiffDecoderPackedBits.initialize(
                    page, 
                    decompress, 
                    unpredict, 
                    page.fillorder, 
                    dtype, 
                    page.bitspersample, 
                    page.imagewidth * page.samplesperpixel
                )

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        raise NotImplementedError("Subclass must implement get_instance")

cdef class TiffDecoderError(TiffDecoder):
    """Decoder that raises an error."""
    
    @staticmethod
    cdef TiffDecoderError initialize(TiffPage page, str error_message):
        cdef TiffDecoderError decoder = TiffDecoderError(page)
        decoder.error_message = error_message
        return decoder

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderErrorInstance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderJpeg(TiffDecoder):
    """Decoder for JPEG compressed segments."""
    
    @staticmethod
    cdef TiffDecoderJpeg initialize(TiffPage page, object colorspace, object outcolorspace):
        cdef TiffDecoderJpeg decoder = TiffDecoderJpeg(page)
        decoder.colorspace = colorspace
        decoder.outcolorspace = outcolorspace
        return decoder

    def __call__(self, FileHandle fh, int64_t offset, int64_t read_len, int64_t index, **kwargs):
        cdef bint _fullsize = kwargs.get('_fullsize', False)
        cdef object jpegtables = kwargs.get('jpegtables', None)
        cdef object jpegheader = kwargs.get('jpegheader', None)
        cdef tuple segmentindex, shape
        cdef object data = None if read_len <= 0 else fh.read_array(numpy.uint8, offset=offset, count=read_len)
        
        segmentindex, shape = self.get_indices_shape(index)
        if data is None:
            if _fullsize:
                shape = self.pad_none(shape)
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
        data_array = self.reshape_data(data_array, segmentindex, shape)
        if _fullsize:
            data_array, shape = self.pad_data(data_array, shape)
        return data_array, segmentindex, shape

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderJpegInstance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderEer(TiffDecoder):
    """Decoder for EER compressed segments."""
    
    @staticmethod
    cdef TiffDecoderEer initialize(TiffPage page, object decompress, int64_t rlebits, int64_t horzbits, int64_t vertbits):
        cdef TiffDecoderEer decoder = TiffDecoderEer(page)
        decoder.decompress = decompress
        decoder.rlebits = rlebits
        decoder.horzbits = horzbits
        decoder.vertbits = vertbits
        return decoder

    def __call__(self, FileHandle fh, int64_t offset, int64_t read_len, int64_t index, **kwargs):
        cdef bint _fullsize = kwargs.get('_fullsize', False)
        cdef tuple segmentindex, shape
        cdef object data = None if read_len <= 0 else fh.read_array(numpy.uint8, offset=offset, count=read_len)
        
        segmentindex, shape = self.get_indices_shape(index)
        if data is None:
            if _fullsize:
                shape = self.pad_none(shape)
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

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderEerInstance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderJetraw(TiffDecoder):
    """Decoder for Jetraw compressed segments."""
    
    @staticmethod
    cdef TiffDecoderJetraw initialize(TiffPage page, object decompress):
        cdef TiffDecoderJetraw decoder = TiffDecoderJetraw(page)
        decoder.decompress = decompress
        return decoder

    def __call__(self, FileHandle fh, int64_t offset, int64_t read_len, int64_t index, **kwargs):
        cdef bint _fullsize = kwargs.get('_fullsize', False)
        cdef tuple segmentindex, shape
        cdef object data = None if read_len <= 0 else fh.read_array(numpy.uint8, offset=offset, count=read_len)
        
        segmentindex, shape = self.get_indices_shape(index)
        if data is None:
            if _fullsize:
                shape = self.pad_none(shape)
            return data, segmentindex, shape
            
        data_array = numpy.zeros(shape, numpy.uint16)
        self.decompress(data, out=data_array)
        return data_array, segmentindex, shape

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderJetrawInstance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderImage(TiffDecoder):
    """Decoder for image compressions."""
    
    @staticmethod
    cdef TiffDecoderImage initialize(TiffPage page, object decompress):
        cdef TiffDecoderImage decoder = TiffDecoderImage(page)
        decoder.decompress = decompress
        return decoder

    def __call__(self, FileHandle fh, int64_t offset, int64_t read_len, int64_t index, **kwargs):
        cdef bint _fullsize = kwargs.get('_fullsize', False)
        cdef tuple segmentindex, shape
        cdef object data = None if read_len <= 0 else fh.read_array(numpy.uint8, offset=offset, count=read_len)
        
        segmentindex, shape = self.get_indices_shape(index)
        if data is None:
            if _fullsize:
                shape = self.pad_none(shape)
            return data, segmentindex, shape
            
        data_array = self.decompress(data)
        data_array = self.reshape_data(data_array, segmentindex, shape)
        if _fullsize:
            data_array, shape = self.pad_data(data_array, shape)
        return data_array, segmentindex, shape

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderImageInstance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderBase(TiffDecoder):
    """Base class for other format decoders."""
    
    def __call__(self, FileHandle fh, int64_t offset, int64_t read_len, int64_t index, **kwargs):
        cdef bint _fullsize = kwargs.get('_fullsize', False)
        cdef tuple segmentindex, shape
        cdef int64_t size
        cdef object data = None if read_len <= 0 else fh.read_array(numpy.uint8, offset=offset, count=read_len)
        
        segmentindex, shape = self.get_indices_shape(index)
        if data is None:
            if _fullsize:
                shape = self.pad_none(shape)
            return data, segmentindex, shape
            
        if self.fillorder == 2:
            data = imagecodecs.bitorder_decode(data)
        if self.decompress is not None:
            size = shape[0] * shape[1] * shape[2] * shape[3]
            data = self.decompress(data, out=size * self.dtype.itemsize)
            
        data_array = self.unpack(data)
        data_array = self.reshape_data(data_array, segmentindex, shape)
        data_array = data_array.astype('=' + self.dtype.char, copy=False)
        if self.unpredict is not None:
            data_array = self.unpredict(data_array, axis=-2, out=data_array)
        if _fullsize:
            data_array, shape = self.pad_data(data_array, shape)
        return data_array, segmentindex, shape
    
    cdef object unpack(self, object data):
        """Unpack data."""
        raise NotImplementedError("Subclass must implement unpack")

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderBaseInstance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderComplexInt(TiffDecoderBase):
    """Decoder for complex integers."""
    
    @staticmethod
    cdef TiffDecoderComplexInt initialize(TiffPage page, object decompress, object unpredict, int64_t fillorder, object dtype, object itype, object ftype):
        cdef TiffDecoderComplexInt decoder = TiffDecoderComplexInt(page)
        decoder.decompress = decompress
        decoder.unpredict = unpredict
        decoder.fillorder = fillorder
        decoder.dtype = dtype
        decoder.itype = itype
        decoder.ftype = ftype
        return decoder
        
    cdef object unpack(self, object data):
        """Return complex integer as numpy.complex."""
        return numpy.frombuffer(data, self.itype).astype(self.ftype).view(self.dtype)

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderComplexIntInstance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderRegular(TiffDecoderBase):
    """Decoder for regular data types."""
    
    @staticmethod
    cdef TiffDecoderRegular initialize(TiffPage page, object decompress, object unpredict, int64_t fillorder, object dtype):
        cdef TiffDecoderRegular decoder = TiffDecoderRegular(page)
        decoder.decompress = decompress
        decoder.unpredict = unpredict
        decoder.fillorder = fillorder
        decoder.dtype = dtype
        return decoder

    cdef object unpack(self, object data):
        """Return numpy array from buffer."""
        try:
            # read only numpy array
            return numpy.frombuffer(data, self.dtype)
        except ValueError:
            # for example, LZW strips may be missing EOI
            bps = self.page.bitspersample // 8
            size = (len(data) // bps) * bps
            return numpy.frombuffer(data[:size], self.dtype)

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderRegularInstance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderRGB(TiffDecoderBase):
    """Decoder for RGB packed integers."""
    
    @staticmethod
    cdef TiffDecoderRGB initialize(TiffPage page, object decompress, object unpredict, int64_t fillorder, object dtype, tuple bitspersample_rgb):
        cdef TiffDecoderRGB decoder = TiffDecoderRGB(page)
        decoder.decompress = decompress
        decoder.unpredict = unpredict
        decoder.fillorder = fillorder
        decoder.dtype = dtype
        decoder.bitspersample_rgb = bitspersample_rgb
        return decoder
        
    cdef object unpack(self, object data):
        """Return numpy array from packed integers."""
        return unpack_rgb(data, self.dtype, self.bitspersample_rgb)

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderRGBInstance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderFloat24(TiffDecoderBase):
    """Decoder for float24 data type."""
    
    @staticmethod
    cdef TiffDecoderFloat24 initialize(TiffPage page, object decompress, object unpredict, int64_t fillorder, object dtype):
        cdef TiffDecoderFloat24 decoder = TiffDecoderFloat24(page)
        decoder.decompress = decompress
        decoder.unpredict = unpredict
        decoder.fillorder = fillorder
        decoder.dtype = dtype
        return decoder

    cdef object unpack(self, object data):
        """Return numpy.float32 array from float24."""
        return imagecodecs.float24_decode(
            data, byteorder=self.page.tiff.byteorder_str
        )

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderFloat24Instance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderPackedBits(TiffDecoderBase):
    """Decoder for bilevel and packed integers."""
    
    @staticmethod
    cdef TiffDecoderPackedBits initialize(TiffPage page, object decompress, object unpredict, int64_t fillorder, object dtype, int64_t bitspersample, int64_t runlen):
        cdef TiffDecoderPackedBits decoder = TiffDecoderPackedBits(page)
        decoder.decompress = decompress
        decoder.unpredict = unpredict
        decoder.fillorder = fillorder
        decoder.dtype = dtype
        decoder.bitspersample = bitspersample
        decoder.runlen = runlen
        return decoder
        
    cdef object unpack(self, object data):
        """Return NumPy array from packed integers."""
        return imagecodecs.packints_decode(
            data, self.dtype, self.bitspersample, runlen=self.runlen
        )

    cdef TiffDecoderInstance get_instance(self, object fh, vector[int64_t] offsets, vector[int64_t] read_lengths, vector[int64_t] indices, object output_fun, dict kwargs):
        """Create an instance for batch processing."""
        return TiffDecoderInstance.create(TiffDecoderPackedBitsInstance, self, fh, offsets, read_lengths, indices, output_fun, kwargs)

cdef class TiffDecoderInstance:
    """Base class for batch decoder instances."""
    @staticmethod
    cdef TiffDecoderInstance create(cls, TiffDecoder decoder, FileHandle fh, vector[int64_t] offsets, 
                vector[int64_t] read_lengths, vector[int64_t] indices, 
                object output_fun, dict kwargs):
        cdef TiffDecoderInstance self = cls()
        self.decoder = decoder
        self.fh = fh
        self.offsets = offsets
        self.read_lengths = read_lengths
        cdef int64_t read_len, max_read_len = 0
        for read_len in read_lengths:
            if read_len > max_read_len:
                max_read_len = read_len
        self.max_read_len = max_read_len
        self.indices = indices
        self.output_fun = output_fun
        if not callable(output_fun):
            raise ValueError("output_fun must be callable")
        self.kwargs = kwargs
        return self
    
    def __call__(self):
        """Process all items in batch."""
        raise NotImplementedError("Subclass must implement __call__")

cdef class TiffDecoderErrorInstance(TiffDecoderInstance):
    """Instance for error decoder."""
    
    def __call__(self):
        """Process all items in batch."""
        cdef TiffDecoderError decoder = <TiffDecoderError>self.decoder
        cdef size_t i
        cdef size_t n = self.indices.size()
        
        for i in range(n):
            raise TiffFileError(decoder.error_message)

cdef class TiffDecoderJpegInstance(TiffDecoderInstance):
    """Instance for JPEG compressed segments."""

    def __call__(self):
        """Process all items in batch."""
        cdef TiffDecoderJpeg decoder = <TiffDecoderJpeg>self.decoder
        cdef size_t i
        cdef size_t n = self.indices.size()
        cdef int64_t index, offset, read_len
        cdef tuple segmentindex, shape
        cdef object data
        cdef bint _fullsize = self.kwargs.get('_fullsize', False)
        cdef object jpegtables = self.kwargs.get('jpegtables', None)
        cdef object jpegheader = self.kwargs.get('jpegheader', None)
        
        for i in range(n):
            index = self.indices[i]
            offset = self.offsets[i]
            read_len = self.read_lengths[i]
            
            segmentindex, shape = decoder.get_indices_shape(index)
            
            if read_len <= 0:
                if _fullsize:
                    shape = decoder.pad_none(shape)
                self.output_fun(None, segmentindex, shape)
                continue
                
            data = self.fh.read_array(numpy.uint8, offset=offset, count=read_len)
            
            data_array = imagecodecs.jpeg_decode(
                data,
                bitspersample=decoder.page.bitspersample,
                tables=self.jpegtables,
                header=self.jpegheader,
                colorspace=decoder.colorspace,
                outcolorspace=decoder.outcolorspace,
                shape=shape[1:3]
            )
            
            data_array = decoder.reshape_data(data_array, segmentindex, shape)
            
            if _fullsize:
                data_array, shape = decoder.pad_data(data_array, shape)
            
            self.output_fun(data_array, segmentindex, shape)

cdef class TiffDecoderEerInstance(TiffDecoderInstance):
    """Instance for EER compressed segments."""
    
    def __call__(self):
        """Process all items in batch."""
        cdef TiffDecoderEer decoder = <TiffDecoderEer>self.decoder
        cdef size_t i
        cdef size_t n = self.indices.size()
        cdef int64_t index, offset, read_len
        cdef tuple segmentindex, shape
        cdef object data
        cdef bint _fullsize = self.kwargs.get('_fullsize', False)
        
        for i in range(n):
            index = self.indices[i]
            offset = self.offsets[i]
            read_len = self.read_lengths[i]
            
            segmentindex, shape = decoder.get_indices_shape(index)
            
            if read_len <= 0:
                if _fullsize:
                    shape = decoder.pad_none(shape)
                self.output_fun(None, segmentindex, shape)
                continue
                
            data = self.fh.read_array(numpy.uint8, offset=offset, count=read_len)
            
            data_array = decoder.decompress(
                data,
                shape=shape[1:3],
                rlebits=decoder.rlebits,
                horzbits=decoder.horzbits,
                vertbits=decoder.vertbits,
                superres=False
            )
            
            data_array = data_array.reshape(shape)
            
            self.output_fun(data_array, segmentindex, shape)

cdef class TiffDecoderJetrawInstance(TiffDecoderInstance):
    """Instance for Jetraw compressed segments."""
    
    def __call__(self):
        """Process all items in batch."""
        cdef TiffDecoderJetraw decoder = <TiffDecoderJetraw>self.decoder
        cdef size_t i
        cdef size_t n = self.indices.size()
        cdef int64_t index, offset, read_len
        cdef tuple segmentindex, shape
        cdef object data
        cdef bint _fullsize = self.kwargs.get('_fullsize', False)
        
        for i in range(n):
            index = self.indices[i]
            offset = self.offsets[i]
            read_len = self.read_lengths[i]
            
            segmentindex, shape = decoder.get_indices_shape(index)
            
            if read_len <= 0:
                if _fullsize:
                    shape = decoder.pad_none(shape)
                self.output_fun(None, segmentindex, shape)
                continue
                
            data = self.fh.read_array(numpy.uint8, offset=offset, count=read_len)
            
            data_array = numpy.zeros(shape, numpy.uint16)
            decoder.decompress(data, out=data_array)
            
            self.output_fun(data_array, segmentindex, shape)

cdef class TiffDecoderImageInstance(TiffDecoderInstance):
    """Instance for image compressions."""
    
    def __call__(self):
        """Process all items in batch."""
        cdef TiffDecoderImage decoder = <TiffDecoderImage>self.decoder
        cdef size_t i
        cdef size_t n = self.indices.size()
        cdef int64_t index, offset, read_len
        cdef tuple segmentindex, shape
        cdef object data
        cdef bint _fullsize = self.kwargs.get('_fullsize', False)
        
        for i in range(n):
            index = self.indices[i]
            offset = self.offsets[i]
            read_len = self.read_lengths[i]
            
            segmentindex, shape = decoder.get_indices_shape(index)
            
            if read_len <= 0:
                if _fullsize:
                    shape = decoder.pad_none(shape)
                self.output_fun(None, segmentindex, shape)
                continue
                
            data = self.fh.read_array(numpy.uint8, offset=offset, count=read_len)
            
            data_array = decoder.decompress(data)
            data_array = decoder.reshape_data(data_array, segmentindex, shape)
            
            if _fullsize:
                data_array, shape = decoder.pad_data(data_array, shape)
            
            self.output_fun(data_array, segmentindex, shape)

cdef class TiffDecoderBaseInstance(TiffDecoderInstance):
    """Instance for other format decoders."""
    
    def __call__(self):
        """Process all items in batch."""
        cdef TiffDecoderBase decoder = <TiffDecoderBase>self.decoder
        cdef size_t i
        cdef size_t n = self.indices.size()
        cdef int64_t index, offset, read_len, size
        cdef tuple segmentindex, shape
        cdef object data, data_array
        cdef bint _fullsize = self.kwargs.get('_fullsize', False)

        cdef object read_buf = numpy.empty((self.max_read_len), dtype=numpy.uint8)
        cdef uint8_t[::1] read_buf_view = read_buf
        
        for i in range(n):
            index = self.indices[i]
            offset = self.offsets[i]
            read_len = self.read_lengths[i]
            
            segmentindex, shape = decoder.get_indices_shape(index)

            with nogil:
                if read_len <= 0 or self.fh.read_into(&read_buf_view[0], offset, read_len) != read_len:
                    with gil:
                        if _fullsize:
                            shape = decoder.pad_none(shape)
                        self.output_fun(None, segmentindex, shape)
                    continue
            
            data = read_buf[:read_len]
            if decoder.fillorder == 2:
                data = imagecodecs.bitorder_decode(data)
                
            if decoder.decompress is not None:
                size = shape[0] * shape[1] * shape[2] * shape[3]
                data = decoder.decompress(data, out=size * decoder.dtype.itemsize)
                
            data_array = decoder.unpack(data)
            data_array = decoder.reshape_data(data_array, segmentindex, shape)
            data_array = data_array.astype('=' + decoder.dtype.char, copy=False)
            
            if decoder.unpredict is not None:
                data_array = decoder.unpredict(data_array, axis=-2, out=data_array)
                
            if _fullsize:
                data_array, shape = decoder.pad_data(data_array, shape)
                
            self.output_fun(data_array, segmentindex, shape)

# For the specialized base decoder instances, we can reuse the TiffDecoderBaseInstance since the main
# difference is in the unpack method which is already handled in the base TiffDecoder classes
cdef class TiffDecoderComplexIntInstance(TiffDecoderBaseInstance):
    """Instance for complex integers."""
    pass

cdef class TiffDecoderRegularInstance(TiffDecoderBaseInstance):
    """Instance for regular data types."""
    pass

cdef class TiffDecoderRGBInstance(TiffDecoderBaseInstance):
    """Instance for RGB packed integers."""
    pass

cdef class TiffDecoderFloat24Instance(TiffDecoderBaseInstance):
    """Instance for float24 data type."""
    pass

cdef class TiffDecoderPackedBitsInstance(TiffDecoderBaseInstance):
    """Instance for bilevel and packed integers."""
    pass
