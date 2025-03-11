#cython: language_level=3
#cython: boundscheck=True
#cython: wraparound=True
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free

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

from concurrent.futures import ThreadPoolExecutor
cimport cython
import math
import numpy
import os
import struct
import warnings

from .decoder cimport TiffDecoder
from .decoder import PREDICTORS, UNPREDICTORS, COMPRESSORS, DECOMPRESSORS


try:
    import imagecodecs
except ImportError:
    # load pure Python implementation of some codecs
    try:
        from . import _imagecodecs as imagecodecs  # type: ignore[no-redef]
    except ImportError:
        import _imagecodecs as imagecodecs  # type: ignore[no-redef]

CACHED_DECODERS = dict()

def imagej_metadata(
    data: bytes, bytecounts: Sequence[int], byteorder_str
) -> dict[str, Any]:
    """Return IJMetadata tag value.

    Parameters:
        bytes:
            Encoded value of IJMetadata tag.
        bytecounts:
            Value of IJMetadataByteCounts tag.
        byteorder_str:
            Byte order of TIFF file.

    Returns:
        Metadata dict with optional items:

            'Info' (str):
                Human-readable information as string.
                Some formats, such as OIF or ScanImage, can be parsed into
                dicts with :py:func:`matlabstr2py` or the
                `oiffile.SettingsFile()` function of the
                `oiffile <https://pypi.org/project/oiffile/>`_  package.
            'Labels' (Sequence[str]):
                Human-readable labels for each channel.
            'Ranges' (Sequence[float]):
                Lower and upper values for each channel.
            'LUTs' (list[numpy.ndarray[(3, 256), 'uint8']]):
                Color palettes for each channel.
            'Plot' (bytes):
                Undocumented ImageJ internal format.
            'ROI', 'Overlays' (bytes):
                Undocumented ImageJ internal region of interest and overlay
                format. Can be parsed with the
                `roifile <https://pypi.org/project/roifile/>`_  package.
            'Properties' (dict[str, str]):
                Map of key, value items.

    """

    def _string(data: bytes, byteorder_str) -> str:
        return data.decode('utf-16' + {'>': 'be', '<': 'le'}[byteorder_str])

    def _doubles(data: bytes, byteorder_str) -> tuple[float, ...]:
        cdef int64_t count = len(data) // 8
        return struct.unpack(byteorder_str + ('d' * count), data)

    def _lut(data: bytes, byteorder_str) -> NDArray[cnp.uint8_t]:
        return numpy.frombuffer(data, numpy.uint8).reshape(-1, 256)

    def _bytes(data: bytes, byteorder_str) -> bytes:
        return data

    # big-endian
    metadata_types: dict[
        bytes, tuple[str, Callable[[bytes, byteorder_str], Any]]
    ] = {
        b'info': ('Info', _string),
        b'labl': ('Labels', _string),
        b'rang': ('Ranges', _doubles),
        b'luts': ('LUTs', _lut),
        b'plot': ('Plot', _bytes),
        b'roi ': ('ROI', _bytes),
        b'over': ('Overlays', _bytes),
        b'prop': ('Properties', _string),
    }
    # little-endian
    metadata_types.update({k[::-1]: v for k, v in metadata_types.items()})

    if len(bytecounts) == 0:
        raise ValueError('no ImageJ metadata')

    if data[:4] not in {b'IJIJ', b'JIJI'}:
        raise ValueError('invalid ImageJ metadata')

    header_size = bytecounts[0]
    if header_size < 12 or header_size > 804:
        raise ValueError('invalid ImageJ metadata header size')

    ntypes = (header_size - 4) // 8
    header = struct.unpack(
        byteorder_str + '4sI' * ntypes, data[4 : 4 + ntypes * 8]
    )
    pos = 4 + ntypes * 8
    counter = 0
    result = {}
    for i in range(0, len(header), 2):
        mtype = header[i]
        count = header[i+1]
        values = []
        name, func = metadata_types.get(mtype, (bytes2str(mtype), _bytes))
        for _ in range(count):
            counter += 1
            pos1 = pos + bytecounts[counter]
            values.append(func(data[pos:pos1], byteorder_str))
            pos = pos1
        result[name.strip()] = values[0] if count == 1 else values
    prop = result.get('Properties')
    if prop and len(prop) % 2 == 0:
        result['Properties'] = dict(
            prop[i : i + 2] for i in range(0, len(prop), 2)
        )
    return result

cdef tuple jpeg_shape(bytes jpeg):
    """Return bitdepth and shape of JPEG image."""
    cdef int64_t i = 0
    cdef int64_t marker, length
    
    while i < len(jpeg):
        marker = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if marker == 0xFFD8:
            # start of image
            continue
        if marker == 0xFFD9:
            # end of image
            break
        if 0xFFD0 <= marker <= 0xFFD7:
            # restart marker
            continue
        if marker == 0xFF01:
            # private marker
            continue

        length = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if 0xFFC0 <= marker <= 0xFFC3:
            # start of frame
            return struct.unpack('>BHHB', jpeg[i : i + 6])
        if marker == 0xFFDA:
            # start of scan
            break

        # skip to next marker
        i += length - 2

    raise ValueError('no SOF marker found')

def ndpi_jpeg_tile(jpeg: bytes) -> tuple[int, int, bytes]:
    """Return tile shape and JPEG header from JPEG with restart markers."""
    cdef int64_t marker, length, factor, ncomponents, restartinterval = 0
    cdef int64_t sofoffset = 0, sosoffset = 0
    cdef int64_t mcuwidth = 1, mcuheight = 1, tilelength, tilewidth
    cdef int64_t cid, table
    cdef int64_t i = 0
    
    while i < len(jpeg):
        marker = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if marker == 0xFFD8:
            # start of image
            continue
        if marker == 0xFFD9:
            # end of image
            break
        if 0xFFD0 <= marker <= 0xFFD7:
            # restart marker
            continue
        if marker == 0xFF01:
            # private marker
            continue

        length = struct.unpack('>H', jpeg[i : i + 2])[0]
        i += 2

        if marker == 0xFFDD:
            # define restart interval
            restartinterval = struct.unpack('>H', jpeg[i : i + 2])[0]

        elif marker == 0xFFC0:
            # start of frame
            sofoffset = i + 1
            precision, imlength, imwidth, ncomponents = struct.unpack(
                '>BHHB', jpeg[i : i + 6]
            )
            i += 6
            mcuwidth = 1
            mcuheight = 1
            for _ in range(ncomponents):
                cid, factor, table = struct.unpack('>BBB', jpeg[i : i + 3])
                i += 3
                if factor >> 4 > mcuwidth:
                    mcuwidth = factor >> 4
                if factor & 0b00001111 > mcuheight:
                    mcuheight = factor & 0b00001111
            mcuwidth *= 8
            mcuheight *= 8
            i = sofoffset - 1

        elif marker == 0xFFDA:
            # start of scan
            sosoffset = i + length - 2
            break

        # skip to next marker
        i += length - 2

    if restartinterval == 0 or sofoffset == 0 or sosoffset == 0:
        raise ValueError('missing required JPEG markers')

    # patch jpeg header for tile size
    tilelength = mcuheight
    tilewidth = restartinterval * mcuwidth
    jpegheader = (
        jpeg[:sofoffset]
        + struct.pack('>HH', tilelength, tilewidth)
        + jpeg[sofoffset + 4 : sosoffset]
    )
    return tilelength, tilewidth, jpegheader


@cython.final
cdef class TiffPage:
    """TIFF image file directory (IFD).

    TiffPage instances are not thread-safe. All attributes are read-only.

    Parameters:
        parent:
            TiffFile instance to read page from.
            The file handle position must be at an offset to an IFD structure.
        index:
            Index of page in IFD tree.
        keyframe:
            Not used.

    Raises:
        TiffFileError: Invalid TIFF structure.

    """

    def __cinit__(self):
        self.subfiletype = 0
        self.imagewidth = 0
        self.imagelength = 0
        self.imagedepth = 1
        self.tilewidth = 0
        self.tilelength = 0
        self.tiledepth = 1
        self.samplesperpixel = 1
        self.bitspersample = 1
        self.sampleformat = 1
        self.rowsperstrip = 2**32 - 1
        self.compression = 1
        self.planarconfig = 1
        self.fillorder = 1
        self.photometric = 0
        self.predictor = 1
        self.extrasamples = ()
        self.subsampling = None
        self.subifds = None
        self.jpegtables = None
        self.jpegheader = None
        self.software = ''
        self.description = ''
        self.description1 = ''
        self.nodata = 0

    def __init__(
        self,
        parent: TiffFile,
        index: int | Sequence[int],
        *,
        TiffPage keyframe = None,
    ) -> None:
        cdef TiffTag tag, uic2tag
        cdef TiffFormat tiff = parent.tiff
        cdef int64_t tagno

        self.fh = parent.filehandle
        self.tiff = tiff
        self.shape = ()
        self.shaped = (0, 0, 0, 0, 0)
        self.dtype = self._dtype = None
        self.axes = ''
        self.tags = TiffTags()
        cdef TiffTags tags = self.tags
        self.dataoffsets = ()
        self.databytecounts = ()
        if isinstance(index, int):
            self._index = (index,)
        else:
            self._index = tuple(index)

        # read IFD structure and its tags from file
        cdef FileHandle fh = self.fh
        self.offset = fh.tell()  # offset to this IFD
        try:
            tagno = struct.unpack(
                tiff.tagnoformat, fh.read(tiff.tagnosize)
            )[0]
            if tagno > 4096:
                raise ValueError(f'suspicious number of tags {tagno}')
        except Exception as exc:
            raise TiffFileError(f'corrupted tag list @{self.offset}') from exc

        cdef int64_t tagoffset = self.offset + tiff.tagnosize  # fh.tell()
        cdef int64_t tagsize, tagsize_
        tagsize = tagsize_ = tiff.tagsize

        cdef bytes data = fh.read(tagsize * tagno)
        cdef bytes ext
        if len(data) != tagsize * tagno:
            raise TiffFileError('corrupted IFD structure')
        if tiff.is_ndpi():
            # patch offsets/values for 64-bit NDPI file
            tagsize = 16
            fh.seek(8, os.SEEK_CUR)
            ext = fh.read(4 * tagno)  # high bits
            data = b''.join(
                data[i * 12 : i * 12 + 12] + ext[i * 4 : i * 4 + 4]
                for i in range(tagno)
            )

        tagindex = -tagsize
        for i in range(tagno):
            tagindex += tagsize
            tagdata = data[tagindex : tagindex + tagsize]
            try:
                tag = TiffTag.fromfile(
                    self.fh,
                    self.tiff,
                    offset=tagoffset + i * tagsize_,
                    header=tagdata,
                    validate=True
                )
            except TiffFileError as exc:
                logger().error(f'<TiffTag.fromfile> raised {exc!r:.128}')
                continue
            tags.add(tag)

        if not tags:
            return  # found in FIBICS

        cdef object value
        cdef int64_t code
        cdef str name
        for code, name in TIFF.TAG_ATTRIBUTES.items():
            value = tags.valueof(code)
            if value is None:
                continue
            if code in {270, 305} and not isinstance(value, str):
                # wrong string type for software or description
                continue
            setattr(self, name, value) # TODO: maybe needs __dict__

        value = tags.valueof(270, default=None, index=1)
        if isinstance(value, str):
            self.description1 = value

        if self.subfiletype == 0:
            value = tags.valueof(255)  # SubfileType
            if value == 2:
                self.subfiletype = 0b1  # reduced image
            elif value == 3:
                self.subfiletype = 0b10  # multi-page
        elif not isinstance(self.subfiletype, int):
            # files created by IDEAS
            logger().warning(f'{self!r} invalid {self.subfiletype=}')
            self.subfiletype = 0

        # consolidate private tags; remove them from self.tags
        # if self.is_andor:
        #     self.andor_tags
        # elif self.is_epics:
        #     self.epics_tags
        # elif self.is_ndpi:
        #     self.ndpi_tags
        # if self.is_sis and 34853 in tags:
        #     # TODO: cannot change tag.name
        #     tags[34853].name = 'OlympusSIS2'

        # dataoffsets and databytecounts
        # TileOffsets
        self.dataoffsets = tags.valueof(324)
        if self.dataoffsets is None:
            # StripOffsets
            self.dataoffsets = tags.valueof(273)
            if self.dataoffsets is None:
                # JPEGInterchangeFormat et al.
                self.dataoffsets = tags.valueof(513)
                if self.dataoffsets is None:
                    self.dataoffsets = ()
                    logger().error(f'{self!r} missing data offset tag')
        # TileByteCounts
        self.databytecounts = tags.valueof(325)
        if self.databytecounts is None:
            # StripByteCounts
            self.databytecounts = tags.valueof(279)
            if self.databytecounts is None:
                # JPEGInterchangeFormatLength et al.
                self.databytecounts = tags.valueof(514)

        """
        TODO: this code seems to use undefined variables
        if (
            self.imagewidth == 0
            and self.imagelength == 0
            and self.dataoffsets
            and self.databytecounts
        ):
            # dimensions may be missing in some RAW formats
            # read dimensions from assumed JPEG encoded segment
            try:
                fh.seek(self.dataoffsets[0])
                (
                    precision,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                ) = jpeg_shape(fh.read(min(self.databytecounts[0], 4096)))
            except Exception:
                pass
            else:
                self.imagelength = imagelength
                self.imagewidth = imagewidth
                self.samplesperpixel = samplesperpixel
                if 258 not in tags:
                    self.bitspersample = 8 if precision <= 8 else 16
                if 262 not in tags and samplesperpixel == 3:
                    self.photometric = PHOTOMETRIC.YCBCR
                if 259 not in tags:
                    self.compression = COMPRESSION.OJPEG
                if 278 not in tags:
                    self.rowsperstrip = imagelength

        el
        """
        if self.compression == 6:
            # OJPEG hack. See libtiff v4.2.0 tif_dirread.c#L4082
            if not tags.contains_code(262):
                # PhotometricInterpretation missing
                self.photometric = PHOTOMETRIC.YCBCR
            elif self.photometric == 2:
                # RGB -> YCbCr
                self.photometric = PHOTOMETRIC.YCBCR
            if not tags.contains_code(258):
                # BitsPerSample missing
                self.bitspersample = 8
            if not tags.contains_code(277):
                # SamplesPerPixel missing
                if self.photometric in {2, 6}:
                    self.samplesperpixel = 3
                elif self.photometric in {0, 1}:
                    self.samplesperpixel = 3

        elif self.is_lsm():# or (self.index != 0 and self.parent.is_lsm()):
            # correct non standard LSM bitspersample tags
            tags[258]._fix_lsm_bitspersample()
            if self.compression == 1 and self.predictor != 1:
                # work around bug in LSM510 software
                self.predictor = PREDICTOR.NONE

        elif self.is_vista():# or (self.index != 0 and self.parent.is_vista()):
            # ISS Vista writes wrong ImageDepth tag
            self.imagedepth = 1

        elif self.is_stk():
            # read UIC1tag again now that plane count is known
            tag = tags.get(33628)  # UIC1tag
            assert tag is not None
            fh.seek(tag.valueoffset)
            uic2tag = tags.get(33629)  # UIC2tag
            try:
                tag.value = read_uic1tag(
                    fh,
                    tiff.byteorder,
                    tag.dtype,
                    tag.count,
                    0,
                    planecount=uic2tag.count if uic2tag is not None else 1,
                )
            except Exception as exc:
                logger().warning(
                    f'{self!r} <tifffile.read_uic1tag> raised {exc!r:.128}'
                )

        tag = tags.get(50839)
        if tag is not None:
            # decode IJMetadata tag
            try:
                tag.value = imagej_metadata(
                    tag.value,
                    tags[50838].value,  # IJMetadataByteCounts
                    tiff.byteorder_str,
                )
            except Exception as exc:
                logger().warning(
                    f'{self!r} <tifffile.imagej_metadata> raised {exc!r:.128}'
                )

        # BitsPerSample
        value = tags.valueof(258)
        if value is not None:
            if self.bitspersample != 1:
                pass  # bitspersample was set by ojpeg hack
            elif tags.get(258).count == 1:
                self.bitspersample = int(value)
            else:
                # LSM might list more items than samplesperpixel
                value = value[: self.samplesperpixel]
                if any(v - value[0] for v in value):
                    self.bitspersample = value
                else:
                    self.bitspersample = int(value[0])

        # SampleFormat
        value = tags.valueof(339)
        if value is not None:
            if tags[339].count == 1:
                try:
                    self.sampleformat = SAMPLEFORMAT(value)
                except ValueError:
                    self.sampleformat = int(value)
            else:
                value = value[: self.samplesperpixel]
                if any(v - value[0] for v in value):
                    try:
                        self.sampleformat = SAMPLEFORMAT(value)
                    except ValueError:
                        self.sampleformat = int(value)
                else:
                    try:
                        self.sampleformat = SAMPLEFORMAT(value[0])
                    except ValueError:
                        self.sampleformat = int(value[0])
        elif self.bitspersample == 32 and (
            self.is_indica()# or (self.index != 0 and self.parent.is_indica())
        ):
            # IndicaLabsImageWriter does not write SampleFormat tag
            self.sampleformat = SAMPLEFORMAT.IEEEFP

        if tags.contains_code(322):  # TileWidth
            self.rowsperstrip = 0
        elif tags.contains_code(257):  # ImageLength
            if not tags.contains_code(278) or tags.get(278).count > 1:  # RowsPerStrip
                self.rowsperstrip = self.imagelength
            self.rowsperstrip = min(self.rowsperstrip, self.imagelength)
            # self.stripsperimage = int(math.floor(
            #    float(self.imagelength + self.rowsperstrip - 1) /
            #    self.rowsperstrip))

        # determine dtype
        cdef str dtypestr = TIFF.SAMPLE_DTYPES.get(
            (self.sampleformat, self.bitspersample), None
        )
        if dtypestr is not None:
            dtype = numpy.dtype(dtypestr)
        else:
            dtype = None
        self.dtype = self._dtype = dtype

        # determine shape of data
        cdef int64_t imagelength = self.imagelength
        cdef int64_t imagewidth = self.imagewidth
        cdef int64_t imagedepth = self.imagedepth
        cdef int64_t samplesperpixel = self.samplesperpixel

        if self.photometric == 2 or samplesperpixel > 1:  # PHOTOMETRIC.RGB
            if self.planarconfig == 1:
                self.shaped = (
                    1,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                )
                if imagedepth == 1:
                    self.shape = (imagelength, imagewidth, samplesperpixel)
                    self.axes = 'YXS'
                else:
                    self.shape = (
                        imagedepth,
                        imagelength,
                        imagewidth,
                        samplesperpixel,
                    )
                    self.axes = 'ZYXS'
            else:
                self.shaped = (
                    samplesperpixel,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    1,
                )
                if imagedepth == 1:
                    self.shape = (samplesperpixel, imagelength, imagewidth)
                    self.axes = 'SYX'
                else:
                    self.shape = (
                        samplesperpixel,
                        imagedepth,
                        imagelength,
                        imagewidth,
                    )
                    self.axes = 'SZYX'
        else:
            self.shaped = (1, imagedepth, imagelength, imagewidth, 1)
            if imagedepth == 1:
                self.shape = (imagelength, imagewidth)
                self.axes = 'YX'
            else:
                self.shape = (imagedepth, imagelength, imagewidth)
                self.axes = 'ZYX'

        if not self.databytecounts:
            self.databytecounts = (
                product(self.shape) * (self.bitspersample // 8),
            )
            if self.compression != 1:
                logger().error(f'{self!r} missing ByteCounts tag')

        cdef int64_t maxstrips
        if imagelength and self.rowsperstrip and not self.is_lsm():
            # fix incorrect number of strip bytecounts and offsets
            maxstrips = (
                int(
                    math.floor(imagelength + self.rowsperstrip - 1)
                    / self.rowsperstrip
                )
                * self.imagedepth
            )
            if self.planarconfig == 2:
                maxstrips *= self.samplesperpixel
            if maxstrips != len(self.databytecounts):
                logger().error(
                    f'{self!r} incorrect StripByteCounts count '
                    f'({len(self.databytecounts)} != {maxstrips})'
                )
                self.databytecounts = self.databytecounts[:maxstrips]
            if maxstrips != len(self.dataoffsets):
                logger().error(
                    f'{self!r} incorrect StripOffsets count '
                    f'({len(self.dataoffsets)} != {maxstrips})'
                )
                self.dataoffsets = self.dataoffsets[:maxstrips]

        value = tags.valueof(42113)  # GDAL_NODATA
        if value is not None and dtype is not None:
            try:
                pytype = type(dtype.type(0).item())
                value = value.replace(',', '.')  # comma decimal separator
                self.nodata = pytype(value)
                if not numpy.can_cast(
                    numpy.min_scalar_type(self.nodata), self.dtype
                ):
                    raise ValueError(
                        f'{self.nodata} is not castable to {self.dtype}'
                    )
            except Exception as exc:
                logger().warning(
                    f'{self!r} parsing GDAL_NODATA tag raised {exc!r:.128}'
                )
                self.nodata = 0

        mcustarts = tags.valueof(65426)
        if mcustarts is not None and self.is_ndpi():
            # use NDPI JPEG McuStarts as tile offsets
            mcustarts = mcustarts.astype(numpy.int64)
            high = tags.valueof(65432)
            if high is not None:
                # McuStartsHighBytes
                high = high.astype(numpy.uint64)
                high <<= 32
                mcustarts += high.astype(numpy.int64)
            fh.seek(self.dataoffsets[0])
            jpegheader = fh.read(mcustarts[0])
            try:
                (
                    self.tilelength,
                    self.tilewidth,
                    self.jpegheader,
                ) = ndpi_jpeg_tile(jpegheader)
            except ValueError as exc:
                logger().warning(
                    f'{self!r} <tifffile.ndpi_jpeg_tile> raised {exc!r:.128}'
                )
            else:
                # TODO: optimize tuple(ndarray.tolist())
                databytecounts = numpy.diff(
                    mcustarts, append=self.databytecounts[0]
                )
                self.databytecounts = tuple(
                    databytecounts.tolist()  # type: ignore[arg-type]
                )
                mcustarts += self.dataoffsets[0]
                self.dataoffsets = tuple(mcustarts.tolist())

    @property
    def decode(self):
        """Return decoder for segments.

        The decoder is a callable class instance that decodes image segments 
        with the following signature:

        Parameters:
            data (Union[bytes, None]):
                Encoded bytes of segment (strip or tile) or None for empty
                segments.
            index (int):
                Index of segment in Offsets and Bytecount tag values.
            **kwargs:
                Additional parameters like jpegtables, jpegheader, _fullsize.

        Returns:
            - Decoded segment or None for empty segments.
            - Position of segment in image array of normalized shape
              (separate sample, depth, length, width, contig sample).
            - Shape of segment (depth, length, width, contig samples).
              The shape of strips depends on their linear index.

        Raises:
            ValueError or NotImplementedError:
                Decoding is not supported.
            TiffFileError:
                Invalid TIFF structure.
        """
        global CACHED_DECODERS
        if self.hash in CACHED_DECODERS:
            return CACHED_DECODERS[self.hash]
        
        # Create the decoder using the factory method
        decoder = TiffDecoder.create(self)
        
        # Cache the decoder
        CACHED_DECODERS[self.hash] = decoder
        
        return decoder

    def segments(
        self,
        *,
        lock: threading.RLock | NullContext | None = None,
        maxworkers: int | None = None,
        func: Callable[..., Any] | None = None,  # TODO: type this
        sort: bool = False,
        buffersize: int | None = None,
        _fullsize: bool | None = None,
    ) -> Iterator[
        tuple[
            NDArray[Any] | None,
            tuple[int, int, int, int, int],
            tuple[int, int, int, int],
        ]
    ]:
        """Return iterator over decoded tiles or strips.

        Parameters:
            lock:
                Reentrant lock to synchronize file seeks and reads.
            maxworkers:
                Maximum number of threads to concurrently decode segments.
            func:
                Function to process decoded segment.
            sort:
                Read segments from file in order of their offsets.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.
            _fullsize:
                Internal use.

        Yields:
            - Decoded segment or None for empty segments.
            - Position of segment in image array of normalized shape
              (separate sample, depth, length, width, contig sample).
            - Shape of segment (depth, length, width, contig samples).
              The shape of strips depends on their linear index.

        """
        cdef TiffPage keyframe = self.keyframe  # self or keyframe
        
        if _fullsize is None:
            _fullsize = keyframe.is_tiled()

        cdef dict decodeargs = {'_fullsize': bool(_fullsize)}
        if keyframe.compression in {6, 7, 34892, 33007}:  # JPEG
            decodeargs['jpegtables'] = self.jpegtables
            decodeargs['jpegheader'] = keyframe.jpegheader

        # Get the decoder instance from keyframe
        decoder = keyframe.decode
        
        if func is None:
            # Direct decoding with decoder instance
            process = lambda args: decoder(*args, **decodeargs)
        else:
            # Apply function to decoded result
            process = lambda args: func(decoder(*args, **decodeargs))

        if maxworkers is None or maxworkers < 1:
            maxworkers = keyframe.maxworkers
        if maxworkers < 2:
            for segment in self.fh.read_segments(
                self.dataoffsets,
                self.databytecounts,
                sort=sort,
                buffersize=-1 if buffersize is None else buffersize,
                flat=True,
            ):
                yield process(segment)
        else:
            # reduce memory overhead by processing chunks of up to
            # buffersize of segments because ThreadPoolExecutor.map is not
            # collecting iterables lazily
            with ThreadPoolExecutor(maxworkers) as executor:
                for segments in self.fh.read_segments(
                    self.dataoffsets,
                    self.databytecounts,
                    sort=sort,
                    buffersize=-1 if buffersize is None else buffersize,
                    flat=False,
                ):
                    yield from executor.map(process, segments)

    def asarray(
        self,
        *,
        out: OutputType = None,
        squeeze: bool = True,
        lock: threading.RLock | NullContext | None = None,
        maxworkers: int | None = None,
        buffersize: int | None = None,
    ) -> NDArray[Any]:
        """Return image from page as NumPy array.

        Parameters:
            out:
                Specifies how image array is returned.
                By default, a new NumPy array is created.
                If a *numpy.ndarray*, a writable array to which the image
                is copied.
                If *'memmap'*, directly memory-map the image data in the
                file if possible; else create a memory-mapped array in a
                temporary file.
                If a *string* or *open file*, the file used to create a
                memory-mapped array.
            squeeze:
                Remove all length-1 dimensions (except X and Y) from
                image array.
                If *False*, return the image array with normalized
                5-dimensional shape :py:attr:`TiffPage.shaped`.
            lock:
                Reentrant lock to synchronize seeks and reads from file.
                The default is the lock of the parent's file handle.
            maxworkers:
                Maximum number of threads to concurrently decode segments.
                If *None* or *0*, use up to :py:attr:`_TIFF.MAXWORKERS`
                threads. See remarks in :py:meth:`TiffFile.asarray`.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.

        Returns:
            NumPy array of decompressed, unpredicted, and unpacked image data
            read from Strip/Tile Offsets/ByteCounts, formatted according to
            shape and dtype metadata found in tags and arguments.
            Photometric conversion, premultiplied alpha, orientation, and
            colorimetry corrections are not applied.
            Specifically, CMYK images are not converted to RGB, MinIsWhite
            images are not inverted, color palettes are not applied,
            gamma is not corrected, and CFA images are not demosaciced.
            Exception are YCbCr JPEG compressed images, which are converted to
            RGB.

        Raises:
            ValueError:
                Format of image in file is not supported and cannot be decoded.

        """
        cdef TiffPage keyframe = self.keyframe  # self or keyframe
        cdef object result
        cdef bint closed = False
        
        if 0 in tuple(keyframe.shaped) or keyframe._dtype is None:
            return numpy.empty((0,), keyframe.dtype)

        if len(self.dataoffsets) == 0:
            raise TiffFileError('missing data offset')

        cdef FileHandle fh = self.fh

        if (
            isinstance(out, str)
            and out == 'memmap'
            and keyframe.is_memmappable()
        ):
            # direct memory map array in file
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file', UserWarning
                    )
                    fh.open()
                result = fh.memmap_array(
                    keyframe.tiff.byteorder + keyframe._dtype.char,
                    keyframe.shaped,
                    offset=self.dataoffsets[0],
                )

        elif keyframe.is_contiguous():
            # read contiguous bytes to array
            if keyframe.is_subsampled():
                raise NotImplementedError('chroma subsampling not supported')
            if out is not None:
                out = create_output(out, keyframe.shaped, keyframe._dtype)
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file', UserWarning
                    )
                    fh.open()
                fh.seek(self.dataoffsets[0])
                result = fh.read_array(
                    keyframe.tiff.byteorder_str + keyframe._dtype.char,
                    count=product(keyframe.shaped),
                    offset=0,
                    out=out,
                )
            if keyframe.fillorder == 2:
                result = imagecodecs.bitorder_decode(result, out=result)
            if keyframe.predictor != 1:
                # predictors without compression
                unpredict = UNPREDICTORS[keyframe.predictor]
                if keyframe.predictor == 1:
                    result = unpredict(result, axis=-2, out=result)
                else:
                    # floatpred cannot decode in-place
                    out = unpredict(result, axis=-2, out=result)
                    result[:] = out

        elif (
            keyframe.jpegheader is not None
            and keyframe is self
            and 273 in self.tags  # striped ...
            and self.is_tiled()  # but reported as tiled
            # TODO: imagecodecs can decode larger JPEG
            and self.imagewidth <= 65500
            and self.imagelength <= 65500
        ):
            # decode the whole NDPI JPEG strip
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file', UserWarning
                    )
                    fh.open()
                fh.seek(self.tags[273].value_get()[0])  # StripOffsets
                data = fh.read(self.tags[279].value_get()[0])  # StripByteCounts
            decompress = DECOMPRESSORS[self.compression]
            result = decompress(
                data,
                bitspersample=self.bitspersample,
                out=out,
                # shape=(self.imagelength, self.imagewidth)
            )
            del data

        else:
            # decode individual strips or tiles
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file', UserWarning
                    )
                    fh.open()
                keyframe.decode  # init TiffPage.decode function under lock

            result = create_output(out, keyframe.shaped, keyframe._dtype)

            def func(
                decoderesult: tuple[
                    NDArray[Any] | None,
                    tuple[int, int, int, int, int],
                    tuple[int, int, int, int],
                ],
                keyframe: TiffPage = keyframe,
                out: NDArray[Any] = result,
            ) -> None:
                # copy decoded segments to output array
                segment, (s, d, h, w, _), shape = decoderesult
                if segment is None:
                    out[
                        s, d : d + shape[0], h : h + shape[1], w : w + shape[2]
                    ] = keyframe.nodata
                else:
                    out[
                        s, d : d + shape[0], h : h + shape[1], w : w + shape[2]
                    ] = segment[
                        : keyframe.imagedepth - d,
                        : keyframe.imagelength - h,
                        : keyframe.imagewidth - w,
                    ]
                # except IndexError:
                #     pass  # corrupted file, for example, with too many strips

            for _ in self.segments(
                func=func,
                lock=lock,
                maxworkers=maxworkers,
                buffersize=buffersize,
                sort=True,
                _fullsize=False,
            ):
                pass

        result.shape = keyframe.shaped
        if squeeze:
            try:
                result.shape = keyframe.shape
            except ValueError as exc:
                logger().warning(
                    f'{self!r} <asarray> failed to reshape '
                    f'{result.shape} to {keyframe.shape}, raised {exc!r:.128}'
                )

        if closed:
            # TODO: close file if an exception occurred above
            fh.close()
        return result

    def aszarr(self, **kwargs: Any) -> ZarrTiffStore:
        """Return image from page as Zarr 2 store.

        Parameters:
            **kwarg: Passed to :py:class:`ZarrTiffStore`.

        """
        from .tifffile import ZarrTiffStore
        return ZarrTiffStore(self, **kwargs)

    def asrgb(
        self,
        *,
        uint8: bool = False,
        alpha: Container[int] | None = None,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Return image as RGB(A). Work in progress. Do not use.

        :meta private:

        """
        data = self.asarray(**kwargs)
        keyframe = self.keyframe  # self or keyframe

        if keyframe.photometric == PHOTOMETRIC.PALETTE:
            colormap = keyframe.colormap
            if colormap is None:
                raise ValueError('no colormap')
            if (
                colormap.shape[1] < 2**keyframe.bitspersample
                or keyframe.dtype is None
                or keyframe.dtype.char not in 'BH'
            ):
                raise ValueError('cannot apply colormap')
            if uint8:
                if colormap.max() > 255:
                    colormap >>= 8
                colormap = colormap.astype(numpy.uint8)
            if 'S' in keyframe.axes:
                data = data[..., 0] if keyframe.planarconfig == 1 else data[0]
            data = apply_colormap(data, colormap)

        elif keyframe.photometric == PHOTOMETRIC.RGB:
            if keyframe.extrasamples:
                if alpha is None:
                    alpha = EXTRASAMPLE
                for i, exs in enumerate(keyframe.extrasamples):
                    if exs in EXTRASAMPLE:
                        if keyframe.planarconfig == 1:
                            data = data[..., [0, 1, 2, 3 + i]]
                        else:
                            data = data[:, [0, 1, 2, 3 + i]]
                        break
            else:
                if keyframe.planarconfig == 1:
                    data = data[..., :3]
                else:
                    data = data[:, :3]
            # TODO: convert to uint8?

        elif keyframe.photometric == PHOTOMETRIC.MINISBLACK:
            raise NotImplementedError
        elif keyframe.photometric == PHOTOMETRIC.MINISWHITE:
            raise NotImplementedError
        elif keyframe.photometric == PHOTOMETRIC.SEPARATED:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return data

    def _gettags(
        self,
        codes: Container[int] | None = None,
        lock: threading.RLock | None = None,
    ) -> list[tuple[int, TiffTag]]:
        """Return list of (code, TiffTag)."""
        return [
            (tag.code, tag)
            for tag in self.tags
            if codes is None or tag.code in codes
        ]

    def _nextifd(self) -> int:
        """Return offset to next IFD from file."""
        fh = self.filehandle
        tiff = self.tiff
        fh.seek(self.offset)
        tagno = struct.unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
        fh.seek(self.offset + tiff.tagnosize + tagno * tiff.tagsize)
        return int(
            struct.unpack(tiff.offsetformat, fh.read(tiff.offsetsize))[0]
        )

    def aspage(self) -> TiffPage:
        """Return TiffPage instance."""
        return self

    @property
    def index(self) -> int:
        """Index of page in IFD chain."""
        return self._index[-1]

    @property
    def treeindex(self) -> tuple[int, ...]:
        """Index of page in IFD tree."""
        return self._index

    @property
    def keyframe(self) -> TiffPage:
        """Self."""
        return self

    @keyframe.setter
    def keyframe(self, index: TiffPage) -> None:
        return

    @property
    def name(self) -> str:
        """Name of image array."""
        index = self._index if len(self._index) > 1 else self._index[0]
        return f'TiffPage {index}'

    @property
    def ndim(self) -> int:
        """Number of dimensions in image array."""
        return len(self.shape)

    @property
    def dims(self) -> tuple[str, ...]:
        """Names of dimensions in image array."""
        names = TIFF.AXES_NAMES
        return tuple([names[str(ax)] for ax in self.axes])

    @property
    def sizes(self) -> dict[str, int]:
        """Ordered map of dimension names to lengths."""
        shape = self.shape
        names = TIFF.AXES_NAMES
        return {names[str(ax)]: shape[i] for i, ax in enumerate(self.axes)}

    @property
    def coords(self) -> dict[str, NDArray[Any]]:
        """Ordered map of dimension names to coordinate arrays."""
        if "coords" in self._cache:
            return self._cache["coords"]
        resolution = self.get_resolution()
        coords: dict[str, NDArray[Any]] = {}

        for ax, size in zip(self.axes, self.shape):
            name = TIFF.AXES_NAMES[ax]
            value = None
            step: int | float = 1

            if ax == 'X':
                step = resolution[0]
            elif ax == 'Y':
                step = resolution[1]
            elif ax == 'S':
                value = self._sample_names()
            elif ax == 'Z':
                # a ZResolution tag doesn't exist.
                # use XResolution if it agrees with YResolution
                if resolution[0] == resolution[1]:
                    step = resolution[0]

            if value is not None:
                coords[name] = numpy.asarray(value)
            elif step == 0 or step == 1 or size == 0:
                coords[name] = numpy.arange(size)
            else:
                coords[name] = numpy.linspace(
                    0, size / step, size, endpoint=False, dtype=numpy.float32
                )
            assert len(coords[name]) == size
        self._cache["coords"] = coords
        return coords

    @property
    def attr(self) -> dict[str, Any]:
        """Arbitrary metadata associated with image array."""
        # TODO: what to return?
        return {}

    @property
    def size(self) -> int:
        """Number of elements in image array."""
        return product(self.shape)

    @property
    def nbytes(self) -> int:
        """Number of bytes in image array."""
        if self.dtype is None:
            return 0
        return self.size * self.dtype.itemsize

    @property
    def colormap(self) -> NDArray[numpy.uint16] | None:
        """Value of Colormap tag."""
        return self.tags.valueof(320)

    @property
    def iccprofile(self) -> bytes | None:
        """Value of InterColorProfile tag."""
        return self.tags.valueof(34675)

    @property
    def transferfunction(self) -> NDArray[numpy.uint16] | None:
        """Value of TransferFunction tag."""
        return self.tags.valueof(301)

    def get_resolution(
        self,
        unit: RESUNIT | int | str | None = None,
        scale: float | int | None = None,
    ) -> tuple[int | float, int | float]:
        """Return number of pixels per unit in X and Y dimensions.

        By default, the XResolution and YResolution tag values are returned.
        Missing tag values are set to 1.

        Parameters:
            unit:
                Unit of measurement of returned values.
                The default is the value of the ResolutionUnit tag.
            scale:
                Factor to convert resolution values to meter unit.
                The default is determined from the ResolutionUnit tag.

        """
        scales = {
            1: 1,  # meter, no unit
            2: 100 / 2.54,  # INCH
            3: 100,  # CENTIMETER
            4: 1000,  # MILLIMETER
            5: 1000000,  # MICROMETER
        }
        if unit is not None:
            unit = enumarg(RESUNIT, unit)
            try:
                if scale is None:
                    resolutionunit = self.tags.valueof(296, default=2)
                    scale = scales[resolutionunit]
            except Exception as exc:
                logger().warning(
                    f'{self!r} <get_resolution> raised {exc!r:.128}'
                )
                scale = 1
            else:
                scale2 = scales[unit]
                if scale % scale2 == 0:
                    scale //= scale2
                else:
                    scale /= scale2
        elif scale is None:
            scale = 1

        resolution: list[int | float] = []
        n: int
        d: int
        for code in 282, 283:
            try:
                n, d = self.tags.valueof(code, default=(1, 1))
                if d == 0:
                    value = n * scale
                elif n % d == 0:
                    value = n // d * scale
                else:
                    value = n / d * scale
            except Exception:
                value = 1
            resolution.append(value)
        return resolution[0], resolution[1]

    @property
    def resolution(self) -> tuple[float, float]:
        """Number of pixels per resolutionunit in X and Y directions."""
        if "resolution" in self._cache:
            return self._cache["resolution"]
        # values are returned in (somewhat unexpected) XY order to
        # keep symmetry with the TiffWriter.write resolution argument
        resolution = self.get_resolution()
        result = (float(resolution[0]), float(resolution[1]))
        self._cache["resolution"] = result
        return result

    @property
    def resolutionunit(self) -> int:
        """Unit of measurement for X and Y resolutions."""
        return self.tags.valueof(296, default=2)

    @property
    def datetime(self) -> DateTime | None:
        """Date and time of image creation."""
        value = self.tags.valueof(306)
        if value is None:
            return None
        try:
            return strptime(value)
        except Exception:
            pass
        return None

    @property
    def tile(self) -> tuple[int, ...] | None:
        """Tile depth, length, and width."""
        if not self.is_tiled():
            return None
        if self.tiledepth > 1:
            return (self.tiledepth, self.tilelength, self.tilewidth)
        return (self.tilelength, self.tilewidth)

    @property
    def chunks(self) -> tuple[int, ...]:
        """Shape of images in tiles or strips."""
        if "chunks" in self._cache:
            return self._cache["chunks"]
        cdef list shape = []
        if self.tiledepth > 1:
            shape.append(self.tiledepth)
        if self.is_tiled():
            shape.extend((self.tilelength, self.tilewidth))
        else:
            shape.extend((self.rowsperstrip, self.imagewidth))
        if self.planarconfig == 1 and self.samplesperpixel > 1:
            shape.append(self.samplesperpixel)
        result = tuple(shape)
        self._cache["chunks"] = result
        return result

    @property
    def chunked(self) -> tuple[int, ...]:
        """Shape of chunked image."""
        if "chunked" in self._cache:
            return self._cache["chunked"]
        cdef list shape = []
        if self.planarconfig == 2 and self.samplesperpixel > 1:
            shape.append(self.samplesperpixel)
        if self.is_tiled():
            if self.imagedepth > 1:
                shape.append(
                    (self.imagedepth + self.tiledepth - 1) // self.tiledepth
                )
            shape.append(
                (self.imagelength + self.tilelength - 1) // self.tilelength
            )
            shape.append(
                (self.imagewidth + self.tilewidth - 1) // self.tilewidth
            )
        else:
            if self.imagedepth > 1:
                shape.append(self.imagedepth)
            shape.append(
                (self.imagelength + self.rowsperstrip - 1) // self.rowsperstrip
            )
            shape.append(1)
        if self.planarconfig == 1 and self.samplesperpixel > 1:
            shape.append(1)
        result = tuple(shape)
        self._cache["chunked"] = result
        return result

    cpdef int hash(self):
        """Checksum to identify pages in same series.

        Pages with the same hash can use the same decode function.
        The hash is calculated from the following properties:
        :py:attr:`TiffFile.tiff`,
        :py:attr:`TiffPage.shaped`,
        :py:attr:`TiffPage.rowsperstrip`,
        :py:attr:`TiffPage.tilewidth`,
        :py:attr:`TiffPage.tilelength`,
        :py:attr:`TiffPage.tiledepth`,
        :py:attr:`TiffPage.sampleformat`,
        :py:attr:`TiffPage.bitspersample`,
        :py:attr:`TiffPage.fillorder`,
        :py:attr:`TiffPage.predictor`,
        :py:attr:`TiffPage.compression`,
        :py:attr:`TiffPage.extrasamples`, and
        :py:attr:`TiffPage.photometric`.

        """
        return hash(
            tuple(self.shaped)
            + (
                self.tiff,
                self.rowsperstrip,
                self.tilewidth,
                self.tilelength,
                self.tiledepth,
                self.sampleformat,
                self.bitspersample,
                self.fillorder,
                self.predictor,
                self.compression,
                self.extrasamples,
                self.photometric,
            )
        )

    @property
    def pages(self) -> TiffPages | None:
        """Sequence of sub-pages, SubIFDs."""
        if not self.tags.contains_code(330):
            return None
        if "pages" in self._cache:
            return self._cache["pages"]
        from .tifffile import TiffPages
        result = TiffPages(self, index=self.index)
        self._cache["pages"] = result
        return result

    @property
    def maxworkers(self) -> int:
        """Maximum number of threads for decoding segments.

        A value of 0 disables multi-threading also when stacking pages.

        """
        if self.is_contiguous() or self.dtype is None:
            return 0
        if self.compression in TIFF.IMAGE_COMPRESSIONS:
            return min(TIFF.MAXWORKERS, len(self.dataoffsets))
        bytecount = product(self.chunks) * self.dtype.itemsize
        if bytecount < 2048:
            # disable multi-threading for small segments
            return 0
        if self.compression == 5 and bytecount < 14336:
            # disable multi-threading for small LZW compressed segments
            return 0
        if len(self.dataoffsets) < 4:
            return 1
        if self.compression != 1 or self.fillorder != 1 or self.predictor != 1:
            if imagecodecs is not None:
                return min(TIFF.MAXWORKERS, len(self.dataoffsets))
        return 2  # optimum for large number of uncompressed tiles

    @property
    def is_contiguous(self) -> bool:
        """Image data is stored contiguously.

        Contiguous image data can be read from
        ``offset=TiffPage.dataoffsets[0]`` with ``size=TiffPage.nbytes``.
        Excludes prediction and fillorder.

        """
        if "is_contiguous" in self._cache:
            return self._cache["is_contiguous"]
        if (
            self.sampleformat == 5
            or self.compression != 1
            or self.bitspersample not in {8, 16, 32, 64}
        ):
            self._cache["is_contiguous"] = False
            return False
        if self.tags.contains_code(322):  # TileWidth
            if (
                self.imagewidth != self.tilewidth
                or self.imagelength % self.tilelength
                or self.tilewidth % 16
                or self.tilelength % 16
            ):
                self._cache["is_contiguous"] = False
                return False
            if (
                self.tags.contains_code(32997)  # ImageDepth
                and self.tags.contains_code(32998)  # TileDepth
                and (
                    self.imagelength != self.tilelength
                    or self.imagedepth % self.tiledepth
                )
            ):
                self._cache["is_contiguous"] = False
                return False
        offsets = self.dataoffsets
        bytecounts = self.databytecounts
        if len(offsets) == 0:
            self._cache["is_contiguous"] = False
            return False
        if len(offsets) == 1:
            self._cache["is_contiguous"] = True
            return True
        if self.is_stk() or self.is_lsm():
            self._cache["is_contiguous"] = False
            return True
        if sum(bytecounts) != self.nbytes:
            self._cache["is_contiguous"] = False
            return False
        if all(
            bytecounts[i] != 0 and offsets[i] + bytecounts[i] == offsets[i + 1]
            for i in range(len(offsets) - 1)
        ):
            self._cache["is_contiguous"] = True
            return True
        self._cache["is_contiguous"] = False
        return False

    @property
    def is_final(self) -> bool:
        """Image data are stored in final form. Excludes byte-swapping."""
        return (
            self.is_contiguous()
            and self.fillorder == 1
            and self.predictor == 1
            and not self.is_subsampled()
        )

    @property
    def is_memmappable(self) -> bool:
        """Image data in file can be memory-mapped to NumPy array."""
        return (
            self.fh.is_file
            and self.is_final()
            # and (self.bitspersample == 8 or self.parent.isnative)
            # aligned?
            and self.dtype is not None
            and self.dataoffsets[0] % self.dtype.itemsize == 0
        )

    def __repr__(self) -> str:
        index = self._index if len(self._index) > 1 else self._index[0]
        return f'<tifffile.TiffPage {index} @{self.offset}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string containing information about TiffPage."""
        if self.keyframe != self:
            from .tifffile import TiffFrame
            return TiffFrame._str(
                self, detail, width  # type: ignore[arg-type]
            )
        attr = ''
        for name in ('memmappable', 'final', 'contiguous'):
            attr = getattr(self, 'is_' + name)()
            if attr:
                attr = name.upper()
                break

        def tostr(name: str, skip: int = 1) -> str:
            obj = getattr(self, name)
            if obj == skip:
                return ''
            try:
                value = getattr(obj, 'name')
            except AttributeError:
                return ''
            return str(value)

        info = '  '.join(
            s.lower()
            for s in (
                'x'.join(str(i) for i in self.shape),
                f'{SAMPLEFORMAT(self.sampleformat).name}{self.bitspersample}',
                ' '.join(
                    i
                    for i in (
                        PHOTOMETRIC(self.photometric).name,
                        'REDUCED' if self.is_reduced() else '',
                        'MASK' if self.is_mask() else '',
                        'TILED' if self.is_tiled() else '',
                        tostr('compression'),
                        tostr('planarconfig'),
                        tostr('predictor'),
                        tostr('fillorder'),
                    )
                    + (attr,)
                    if i
                ),
                '|'.join(f.upper() for f in sorted(self.flags)),
            )
            if s
        )
        index = self._index if len(self._index) > 1 else self._index[0]
        info = f'TiffPage {index} @{self.offset}  {info}'
        if detail <= 0:
            return info
        info_list = [info, self.tags._str(detail + 1, width=width)]
        if detail > 1:
            for name in ('ndpi',):
                name = name + '_tags'
                attr = getattr(self, name, '')
                if attr:
                    info_list.append(
                        f'{name.upper()}\n'
                        f'{pformat(attr, width=width, height=detail * 8)}'
                    )
        if detail > 3:
            try:
                data = self.asarray()
                info_list.append(
                    f'DATA\n'
                    f'{pformat(data, width=width, height=detail * 8)}'
                )
            except Exception:
                pass
        return '\n\n'.join(info_list)

    def _sample_names(self) -> list[str] | None:
        """Return names of samples."""
        if 'S' not in self.axes:
            return None
        samples = self.shape[self.axes.find('S')]
        extrasamples = len(self.extrasamples)
        if samples < 1 or extrasamples > 2:
            return None
        if self.photometric == 0:
            names = ['WhiteIsZero']
        elif self.photometric == 1:
            names = ['BlackIsZero']
        elif self.photometric == 2:
            names = ['Red', 'Green', 'Blue']
        elif self.photometric == 5:
            names = ['Cyan', 'Magenta', 'Yellow', 'Black']
        elif self.photometric == 6:
            if self.compression in {6, 7, 34892, 33007}:
                # YCBCR -> RGB for JPEG
                names = ['Red', 'Green', 'Blue']
            else:
                names = ['Luma', 'Cb', 'Cr']
        else:
            return None
        if extrasamples > 0:
            names += [enumarg(EXTRASAMPLE, self.extrasamples[0]).name.title()]
        if extrasamples > 1:
            names += [enumarg(EXTRASAMPLE, self.extrasamples[1]).name.title()]
        if len(names) != samples:
            return None
        return names

    @property
    def flags(self) -> set[str]:
        r"""Set of ``is\_\*`` properties that are True."""
        if "flags" in self._cache:
            return self._cache["flags"]
        result = {
            name.lower()
            for name in TIFF.PAGE_FLAGS
            if getattr(self, 'is_' + name)()
        }
        self._cache["flags"] = result
        return result

    @property
    def andor_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from Andor tags."""
        cdef TiffTag tag
        if "andor_tags" in self._cache:
            return self._cache["andor_tags"]
        if not self.is_andor():
            self._cache["andor_tags"] = None
            return None
        result = {'Id': self.tags.valueof(4864)}  # AndorId
        for tag in self.tags.values():
            code = tag.code
            if not 4864 < code < 5031:
                continue
            name = tag.name
            name = name[5:] if len(name) > 5 else name
            result[name] = tag.value_get()
            # del self.tags[code]
        self._cache["andor_tags"] = result
        return result

    @property
    def epics_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from EPICS areaDetector tags.

        Use the :py:func:`epics_datetime` function to get a datetime object
        from the epicsTSSec and epicsTSNsec tags.

        """
        if "epics_tags" in self._cache:
            return self._cache["epics_tags"]
        if not self.is_epics():
            self._cache["epics_tags"] = None
            return None
        result = {}
        for tag in self.tags.values():
            code = tag.code
            if not 65000 <= code < 65500:
                continue
            value = tag.value
            if code == 65000:
                # not a POSIX timestamp
                # https://github.com/bluesky/area-detector-handlers/issues/20
                result['timeStamp'] = float(value)
            elif code == 65001:
                result['uniqueID'] = int(value)
            elif code == 65002:
                result['epicsTSSec'] = int(value)
            elif code == 65003:
                result['epicsTSNsec'] = int(value)
            else:
                key, value = value.split(':', 1)
                result[key] = astype(value)
            # del self.tags[code]
        self._cache["epics_tags"] = result
        return result

    @property
    def ndpi_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from Hamamatsu NDPI tags."""
        # TODO: parse 65449 ini style comments
        if "ndpi_tags" in self._cache:
            return self._cache["ndpi_tags"]
        if not self.is_ndpi():
            self._cache["ndpi_tags"] = None
            return None
        cdef TiffTags tags = self.tags
        result = {}
        for name in ('Make', 'Model', 'Software'):
            result[name] = tags[name].value
        for code, name in TIFF.NDPI_TAGS.items():
            if code in tags:
                result[name] = tags[code].value
                # del tags[code]
        if 'McuStarts' in result:
            mcustarts = result['McuStarts']
            if 'McuStartsHighBytes' in result:
                high = result['McuStartsHighBytes'].astype(numpy.uint64)
                high <<= 32
                mcustarts = mcustarts.astype(numpy.uint64)
                mcustarts += high
                del result['McuStartsHighBytes']
            result['McuStarts'] = mcustarts
        self._cache["ndpi_tags"] = result
        return result

    @property
    def geotiff_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from GeoTIFF tags."""
        if "geotiff_tags" in self._cache:
            return self._cache["geotiff_tags"]
        if not self.is_geotiff():
            self._cache["geotiff_tags"] = None
            return None
        cdef TiffTags tags = self.tags

        gkd = tags.valueof(34735)  # GeoKeyDirectoryTag
        if gkd is None or len(gkd) < 2 or gkd[0] != 1:
            logger().warning(f'{self!r} invalid GeoKeyDirectoryTag')
            return {}

        result = {
            'KeyDirectoryVersion': gkd[0],
            'KeyRevision': gkd[1],
            'KeyRevisionMinor': gkd[2],
            # 'NumberOfKeys': gkd[3],
        }
        # deltags = ['GeoKeyDirectoryTag']
        geokeys = TIFF.GEO_KEYS
        geocodes = TIFF.GEO_CODES
        for index in range(gkd[3]):
            try:
                keyid, tagid, count, offset = gkd[
                    4 + index * 4 : index * 4 + 8
                ]
            except Exception as exc:
                logger().warning(
                    f'{self!r} corrupted GeoKeyDirectoryTag '
                    f'raised {exc!r:.128}'
                )
                continue
            if tagid == 0:
                value = offset
            else:
                try:
                    value = tags[tagid].value[offset : offset + count]
                except TiffFileError as exc:
                    logger().warning(
                        f'{self!r} corrupted GeoKeyDirectoryTag {tagid} '
                        f'raised {exc!r:.128}'
                    )
                    continue
                except KeyError as exc:
                    logger().warning(
                        f'{self!r} GeoKeyDirectoryTag {tagid} not found, '
                        f'raised {exc!r:.128}'
                    )
                    continue
                if tagid == 34737 and count > 1 and value[-1] == '|':
                    value = value[:-1]
                value = value if count > 1 else value[0]
            if keyid in geocodes:
                try:
                    value = geocodes[keyid](value)
                except Exception:
                    pass
            try:
                key = geokeys(keyid).name
            except ValueError:
                key = keyid
            result[key] = value

        value = tags.valueof(33920)  # IntergraphMatrixTag
        if value is not None:
            value = numpy.array(value)
            if value.size == 16:
                value = value.reshape((4, 4)).tolist()
            result['IntergraphMatrix'] = value

        value = tags.valueof(33550)  # ModelPixelScaleTag
        if value is not None:
            result['ModelPixelScale'] = numpy.array(value).tolist()

        value = tags.valueof(33922)  # ModelTiepointTag
        if value is not None:
            value = numpy.array(value).reshape((-1, 6)).squeeze().tolist()
            result['ModelTiepoint'] = value

        value = tags.valueof(34264)  # ModelTransformationTag
        if value is not None:
            value = numpy.array(value).reshape((4, 4)).tolist()
            result['ModelTransformation'] = value

        # if 33550 in tags and 33922 in tags:
        #     sx, sy, sz = tags[33550].value  # ModelPixelScaleTag
        #     tiepoints = tags[33922].value  # ModelTiepointTag
        #     transforms = []
        #     for tp in range(0, len(tiepoints), 6):
        #         i, j, k, x, y, z = tiepoints[tp : tp + 6]
        #         transforms.append(
        #             [
        #                 [sx, 0.0, 0.0, x - i * sx],
        #                 [0.0, -sy, 0.0, y + j * sy],
        #                 [0.0, 0.0, sz, z - k * sz],
        #                 [0.0, 0.0, 0.0, 1.0],
        #             ]
        #         )
        #     if len(tiepoints) == 6:
        #         transforms = transforms[0]
        #     result['ModelTransformation'] = transforms

        rpcc = tags.valueof(50844)  # RPCCoefficientTag
        if rpcc is not None:
            result['RPCCoefficient'] = {
                'ERR_BIAS': rpcc[0],
                'ERR_RAND': rpcc[1],
                'LINE_OFF': rpcc[2],
                'SAMP_OFF': rpcc[3],
                'LAT_OFF': rpcc[4],
                'LONG_OFF': rpcc[5],
                'HEIGHT_OFF': rpcc[6],
                'LINE_SCALE': rpcc[7],
                'SAMP_SCALE': rpcc[8],
                'LAT_SCALE': rpcc[9],
                'LONG_SCALE': rpcc[10],
                'HEIGHT_SCALE': rpcc[11],
                'LINE_NUM_COEFF': rpcc[12:33],
                'LINE_DEN_COEFF ': rpcc[33:53],
                'SAMP_NUM_COEFF': rpcc[53:73],
                'SAMP_DEN_COEFF': rpcc[73:],
            }
        self._cache["geotiff_tags"] = result
        return result

    @property
    def shaped_description(self) -> str | None:
        """Description containing array shape if exists, else None."""
        if "shaped_description" in self._cache:
            return self._cache["shaped_description"]
        for description in (self.description, self.description1):
            if not description or '"mibi.' in description:
                self._cache["shaped_description"] = None
                return None
            if description[:1] == '{' and '"shape":' in description:
                self._cache["shaped_description"] = description
                return description
            if description[:6] == 'shape=':
                self._cache["shaped_description"] = description
                return description
        self._cache["shaped_description"] = description
        return description

    @property
    def imagej_description(self) -> str | None:
        """ImageJ description if exists, else None."""
        if "imagej_description" in self._cache:
            return self._cache["imagej_description"]
        for description in (self.description, self.description1):
            if not description:
                self._cache["imagej_description"] = None
                return None
            if description[:7] == 'ImageJ=' or description[:7] == 'SCIFIO=':
                self._cache["imagej_description"] = description
                return description
        self._cache["imagej_description"] = None
        return None

    cpdef bint is_jfif(self):
        """JPEG compressed segments contain JFIF metadata."""
        if (
            self.compression not in {6, 7, 34892, 33007}
            or len(self.dataoffsets) < 1
            or self.dataoffsets[0] == 0
            or len(self.databytecounts) < 1
            or self.databytecounts[0] < 11
        ):
            return False
        data = self.fh.read_at(self.dataoffsets[0] + 6, 4)
        return data == b'JFIF'  # or data == b'Exif'

    cpdef bint is_ndpi(self):
        """Page contains NDPI metadata."""
        return self.tags.contains_code(65420) and self.tags.contains_code(271)

    cpdef bint is_philips(self):
        """Page contains Philips DP metadata."""
        return self.software[:10] == 'Philips DP' and self.description[
            -16:
        ].strip().endswith('</DataObject>')

    cpdef bint is_eer(self):
        """Page contains EER acquisition metadata."""
        return (
            self.tiff.is_bigtiff()
            and self.compression in {1, 65000, 65001, 65002}
            and self.tags.contains_code(65001)
            and self.tags.get(65001).datatype == 7
        )

    cpdef bint is_mediacy(self):
        """Page contains Media Cybernetics Id tag."""
        cdef TiffTag tag = self.tags.get(50288)  # MC_Id
        try:
            return tag is not None and tag.value_get()[:7] == b'MC TIFF'
        except Exception:
            return False

    cpdef bint is_stk(self):
        """Page contains UIC1Tag tag."""
        return self.tags.contains_code(33628)

    cpdef bint is_lsm(self):
        """Page contains CZ_LSMINFO tag."""
        return self.tags.contains_code(34412)

    cpdef bint is_fluoview(self):
        """Page contains FluoView MM_STAMP tag."""
        return self.tags.contains_code(34362)

    cpdef bint is_nih(self):
        """Page contains NIHImageHeader tag."""
        return self.tags.contains_code(43314)

    cpdef bint is_volumetric(self):
        """Page contains SGI ImageDepth tag with value > 1."""
        return self.imagedepth > 1

    cpdef bint is_vista(self):
        """Software tag is 'ISS Vista'."""
        return self.software == 'ISS Vista'

    cpdef bint is_metaseries(self):
        """Page contains MDS MetaSeries metadata in ImageDescription tag."""
        if self.index != 0 or self.software != 'MetaSeries':
            return False
        d = self.description
        return d.startswith('<MetaData>') and d.endswith('</MetaData>')

    cpdef bint is_ome(self):
        """Page contains OME-XML in ImageDescription tag."""
        if self.index != 0 or not self.description:
            return False
        return self.description[-10:].strip().endswith('OME>')

    cpdef bint is_scn(self):
        """Page contains Leica SCN XML in ImageDescription tag."""
        if self.index != 0 or not self.description:
            return False
        return self.description[-10:].strip().endswith('</scn>')

    cpdef bint is_micromanager(self):
        """Page contains MicroManagerMetadata tag."""
        return self.tags.contains_code(51123)

    cpdef bint is_andor(self):
        """Page contains Andor Technology tags 4864-5030."""
        return self.tags.contains_code(4864)

    cpdef bint is_pilatus(self):
        """Page contains Pilatus tags."""
        return self.software[:8] == 'TVX TIFF' and self.description[:2] == '# '

    cpdef bint is_epics(self):
        """Page contains EPICS areaDetector tags."""
        return (
            self.description == 'EPICS areaDetector'
            or self.software == 'EPICS areaDetector'
        )

    cpdef bint is_tvips(self):
        """Page contains TVIPS metadata."""
        return self.tags.contains_code(37706)

    cpdef bint is_fei(self):
        """Page contains FEI_SFEG or FEI_HELIOS tags."""
        return self.tags.contains_code(34680) or self.tags.contains_code(34682)

    cpdef bint is_sem(self):
        """Page contains CZ_SEM tag."""
        return self.tags.contains_code(34118)

    cpdef bint is_svs(self):
        """Page contains Aperio metadata."""
        return self.description[:7] == 'Aperio '

    cpdef bint is_bif(self):
        """Page contains Ventana metadata."""
        try:
            return self.tags.contains_code(700) and (
                # avoid reading XMP tag from file at this point
                # b'<iScan' in self.tags[700].value[:4096]
                'Ventana' in self.software
                or self.software[:17] == 'ScanOutputManager'
                or self.description == 'Label Image'
                or self.description == 'Label_Image'
                or self.description == 'Probability_Image'
            )
        except Exception:
            return False

    cpdef bint is_scanimage(self):
        """Page contains ScanImage metadata."""
        return (
            self.software[:3] == 'SI.'
            or self.description[:6] == 'state.'
            or 'scanimage.SI' in self.description[-256:]
        )

    cpdef bint is_indica(self):
        """Page contains IndicaLabs metadata."""
        return self.software[:21] == 'IndicaLabsImageWriter'

    cpdef bint is_avs(self):
        """Page contains Argos AVS XML metadata."""
        try:
            value = self.tags.valueof(65000)
            return self.tags.contains_code(65000) and value is not None and value[:6] == '<Argos'
        except Exception:
            return False

    cpdef bint is_qpi(self):
        """Page contains PerkinElmer tissue images metadata."""
        # The ImageDescription tag contains XML with a top-level
        # <PerkinElmer-QPI-ImageDescription> element
        return self.software[:15] == 'PerkinElmer-QPI'

    cpdef bint is_geotiff(self):
        """Page contains GeoTIFF metadata."""
        return self.tags.contains_code(34735)  # GeoKeyDirectoryTag

    cpdef bint is_gdal(self):
        """Page contains GDAL metadata."""
        # startswith '<GDALMetadata>'
        return self.tags.contains_code(42112)  # GDAL_METADATA

    cpdef bint is_astrotiff(self):
        """Page contains AstroTIFF FITS metadata."""
        return (
            self.description[:7] == 'SIMPLE '
            and self.description[-3:] == 'END'
        )

    cpdef bint is_streak(self):
        """Page contains Hamamatsu streak metadata."""
        return (
            self.description[:1] == '['
            and '],' in self.description[1:32]
            # and self.tags.get(315, '').value[:19] == 'Copyright Hamamatsu'
        )

    cpdef bint is_dng(self):
        """Page contains DNG metadata."""
        return self.tags.contains_code(50706)  # DNGVersion

    cpdef bint is_tiffep(self):
        """Page contains TIFF/EP metadata."""
        return self.tags.contains_code(37398)  # TIFF/EPStandardID

    cpdef bint is_sis(self):
        """Page contains Olympus SIS metadata."""
        return self.tags.contains_code(33560) or self.tags.contains_code(33471)

    cpdef bint is_frame(self):
        """Object is :py:class:`TiffFrame` instance."""
        return False

    cpdef bint is_virtual(self):
        """Page does not have IFD structure in file."""
        return False

    cpdef bint is_subifd(self):
        """Page is SubIFD of another page."""
        return len(self._index) > 1

    cpdef bint is_reduced(self):
        """Page is reduced image of another image."""
        return bool(self.subfiletype & 0b1)

    cpdef bint is_multipage(self):
        """Page is part of multi-page image."""
        return bool(self.subfiletype & 0b10)

    cpdef bint is_mask(self):
        """Page is transparency mask for another image."""
        return bool(self.subfiletype & 0b100)

    cpdef bint is_mrc(self):
        """Page is part of Mixed Raster Content."""
        return bool(self.subfiletype & 0b1000)

    cpdef bint is_imagej(self):
        """Page contains ImageJ description metadata."""
        return self.imagej_description is not None

    cpdef bint is_shaped(self):
        """Page contains Tifffile JSON metadata."""
        return self.shaped_description is not None

    cpdef bint is_mdgel(self):
        """Page contains MDFileTag tag."""
        return (
            not self.tags.contains_code(37701)  # AgilentBinary
            and self.tags.contains_code(33445)  # MDFileTag
        )

    cpdef bint is_agilent(self):
        """Page contains Agilent Technologies tags."""
        # tag 270 and 285 contain color names
        return self.tags.contains_code(285) and self.tags.contains_code(37701)  # AgilentBinary

    cpdef bint is_tiled(self):
        """Page contains tiled image."""
        return self.tilewidth > 0  # return 322 in self.tags  # TileWidth

    cpdef bint is_subsampled(self):
        """Page contains chroma subsampled image."""
        if self.subsampling is not None:
            return self.subsampling != (1, 1)
        return self.photometric == 6  # YCbCr
        # RGB JPEG usually stored as subsampled YCbCr
        # self.compression == 7
        # and self.photometric == 2
        # and self.planarconfig == 1