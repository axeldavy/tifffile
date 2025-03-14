#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=True
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

from libc.math cimport floor
from libc.stdint cimport int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

from .format cimport ByteOrder, TiffFormat
from .files cimport FileHandle
from .series cimport TiffPages
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

cdef get_max_workers():
    """Default maximum number of threads for de/compressing segments.

    The value of the ``TIFFFILE_NUM_THREADS`` environment variable if set,
    else half the CPU cores up to 32.

    """
    if 'TIFFFILE_NUM_THREADS' in os.environ:
        return max(1, int(os.environ['TIFFFILE_NUM_THREADS']))
    cpu_count: int | None
    try:
        cpu_count = len(
            os.sched_getaffinity(0)  # type: ignore[attr-defined]
        )
    except AttributeError:
        cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return min(32, max(1, cpu_count // 2))

MAXWORKERS = get_max_workers()
THREADPOOL = ThreadPoolExecutor(max_workers=MAXWORKERS)


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

    @staticmethod
    cdef TiffPage from_file(
        FileHandle filehandle,
        TiffFormat tiff,
        int64_t offset
    ):
        """Initialize TiffPage from file.
        
        Parameters:
            parent: TiffFile instance to read page from.
                The file handle position must be at an offset to an IFD structure.
            index: Index of page in IFD tree.
            keyframe: Not used.
            
        Raises:
            TiffFileError: Invalid TIFF structure.
        """
        cdef TiffTag tag, uic2tag
        cdef int64_t tagno
        cdef TiffPage self = TiffPage.__new__(TiffPage)

        # Initialize basic attributes
        self.fh = filehandle
        self.offset = offset
        self.tiff = tiff
        self.shape = ()
        self.shaped = (0, 0, 0, 0, 0)
        self.dtype = self._dtype = None
        self.axes = ''
        self.tags = TiffTags(self.fh, self.tiff)
        #self.dataoffsets = ()
        #self.databytecounts = ()
        self._cache = {}

        cdef int ifd_success
        with nogil:
            # Read IFD structure and tags
            if self._read_ifd_structure() != 0:
                raise TiffFileError('corrupted IFD structure')
            
            # Process common tags (dimensions, format, etc)
            self._process_common_tags()
            
            # Handle special format tags
            self._process_special_format_tags()
        
            # Process dataoffsets and databytecounts
            if self._process_data_pointers() != 0:
                raise TiffFileError('missing required tags')
        
        # Determine image shape and dtype
        self._determine_shape_and_dtype()
        return self

    cdef int _read_ifd_structure(self) noexcept nogil:
        """Read the IFD structure and tags from file."""
        cdef bint is_little_endian = self.tiff.byteorder == ByteOrder.II
        
        cdef int64_t cur_pos = self.offset
        
        # Read tag count more efficiently
        cdef uint8_t[8] tagno_p
        cdef uint16_t tagno = 0
        cdef uint64_t tagno_large = 0
        
        if self.tiff.tagnosize == 2:
            # Most common case (regular TIFF)
            if self.fh.read_into(&tagno_p[0], cur_pos, 2) != 2:
                return -1 #raise ValueError('Could not read tag count')
            cur_pos += 2
                    
            if is_little_endian:
                tagno = (<uint16_t*>tagno_p)[0]
            else:
                tagno = (tagno_p[0] << 8) | tagno_p[1]
        else:
            # BigTIFF
            if self.fh.read_into(&tagno_p[0], cur_pos, 8) != 8:
                return -1 #raise ValueError('Could not read tag count')
            cur_pos += 8
                
            if is_little_endian:
                tagno_large = (<uint64_t*>tagno_p)[0]
            else:
                tagno_large = (((<uint64_t>tagno_p[0]) << 56) |
                               ((<uint64_t>tagno_p[1]) << 48) |
                               ((<uint64_t>tagno_p[2]) << 40) |
                               ((<uint64_t>tagno_p[3]) << 32) |
                               ((<uint64_t>tagno_p[4]) << 24) |
                               ((<uint64_t>tagno_p[5]) << 16) |
                               ((<uint64_t>tagno_p[6]) << 8) |
                                (<uint64_t>tagno_p[7]))
            tagno = <uint16_t>tagno_large
                
        if tagno > 4096:
            return -1 #raise ValueError(f'suspicious number of tags {tagno}')

        cdef int64_t tagoffset = self.offset + self.tiff.tagnosize  # self.fh.tell()
        cdef int64_t tagsize = self.tiff.tagsize

        # Read all tag data at once
        cdef uint8_t *data = <uint8_t*>malloc(tagsize * tagno)
        
        if self.fh.read_into(&data[0], cur_pos, tagsize * tagno) != tagsize * tagno:
            free(data)
            return -1 #raise TiffFileError('corrupted IFD structure')

        cdef bint res = self.tags.load_tags(data, tagno, tagsize)
        free(<void*>data)
        return res

    cdef void _process_common_tags(self) noexcept nogil:
        """Process common TIFF tags and set page attributes."""
        # Process SubfileType
        self.subfiletype = self.tags.valueof_int(254, self.subfiletype, 0)

        # Process dimension tags
        self.imagewidth = self.tags.valueof_int(256, self.imagewidth, 0) # ImageWidth
        self.imagelength = self.tags.valueof_int(257, self.imagelength, 0) # ImageLength
        self.imagedepth = self.tags.valueof_int(32997, self.imagedepth, 0) # ImageDepth

        # Process tile-related tags
        self.tilewidth = self.tags.valueof_int(322, self.tilewidth, 0) # TileWidth
        self.tilelength = self.tags.valueof_int(323, self.tilelength, 0) # TileLength
        self.tiledepth = self.tags.valueof_int(32998, self.tiledepth, 0) # TileDepth

        # Process sample-related tags
        self.samplesperpixel = self.tags.valueof_int(277, self.samplesperpixel, 0) # SamplesPerPixel
        self.compression = self.tags.valueof_int(259, self.compression, 0) # Compression
        self.photometric = self.tags.valueof_int(262, self.photometric, 0) # Photometric
        self.planarconfig = self.tags.valueof_int(284, self.planarconfig, 0) # PlanarConfig
        self.fillorder = self.tags.valueof_int(266, self.fillorder, 0) # FillOrder
        self.predictor = self.tags.valueof_int(317, self.predictor, 0) # Predictor
        self.rowsperstrip = self.tags.valueof_int(278, self.rowsperstrip, 0) # RowsPerStrip

        # TileWidth vs RowsPerStrip handling
        if self.tags.contains_code(322):  # TileWidth
            self.rowsperstrip = 0
        elif self.tags.contains_code(257):  # ImageLength
            if not self.tags.contains_code(278) or self.tags.get_count(278, 0) > 1:  # RowsPerStrip
                self.rowsperstrip = self.imagelength
            self.rowsperstrip = min(self.rowsperstrip, self.imagelength)

        # tags that are processed as python objects require the gil
        if self.tags.contains_code(338) or self.tags.contains_code(530) or\
           self.tags.contains_code(330) or self.tags.contains_code(347) or\
           self.tags.contains_code(270) or self.tags.contains_code(305) or\
           (self.subfiletype == 0 and self.tags.contains_code(255)):
            with gil:
                self._process_common_tags_gil()

    cdef void _process_common_tags_gil(self):
        """Process common TIFF tags and set page attributes."""
        cdef TiffTags tags = self.tags
        cdef object value
        # These are tuples
        self.extrasamples = tags.valueof(338, self.extrasamples, 0) # ExtraSamples
        self.subsampling = tags.valueof(530, self.subsampling, 0) # YCbCrSubSampling
        self.subifds = tags.valueof(330, self.subifds, 0) # SubIFDs

        # Other tags
        self.jpegtables = tags.valueof(347, self.jpegtables, 0) # JPEGTables
        self.description = tags.valueof(270, self.description, 0) # Description
        self.software = tags.valueof(305, self.software, 0) # Software
        self.description1 = tags.valueof(270, self.description1, 1) # Description1

        # Process SubfileType (legacy tag)
        if self.subfiletype == 0:
            value = tags.valueof(255, None, 0) # SubfileType
            if value == 2:
                self.subfiletype = 0b1 # reduced image
            elif value == 3:
                self.subfiletype = 0b10 # multi-page

    cdef void _process_special_format_tags(self) noexcept nogil:
        """Process tags specific to special file formats."""
        # STK (MetaMorph) format
        if self.tags.contains_code(33628):
            with gil:
                self._process_special_format_tags_stk()

        # ImageJ metadata
        if self.tags.contains_code(50839):
            with gil:
                self._process_special_format_tags_imagej()

        # BitsPerSample
        if self.bitspersample == 1 and self.tags.contains_code(258): # not OJPEG hack
            if self.tags.get_count(258, 0) == 1:
                self.bitspersample = self.tags.valueof_int(258, self.bitspersample, 0)
            else:
                with gil:
                    self._process_special_format_tags_bitspersample_2()

        # SampleFormat
        if self.tags.contains_code(339):
            if self.tags.get_count(339, 0) == 1:
                self.sampleformat = self.tags.valueof_int(339, self.sampleformat, 0)
            else:
                with gil:
                    self._process_special_format_tags_sampleformat_2()
        elif self.bitspersample == 32:
            with gil:
                if self.is_indica():
                    # IndicaLabsImageWriter does not write SampleFormat tag
                    self.sampleformat = SAMPLEFORMAT.IEEEFP

        # GDAL NoData tag
        if self.tags.contains_code(42113):
            with gil:
                self._process_special_format_tags_gdal()

    cdef void _process_special_format_tags_stk(self):
        cdef TiffTag tag
        # read UIC1tag again now that plane count is known
        tag = self.tags.get(33628)  # UIC1tag
        assert tag is not None
        self.fh.seek(tag.valueoffset)
        uic2tag = self.tags.get(33629)  # UIC2tag
        try:
            tag.value = read_uic1tag(
                self.fh,
                self.tiff.byteorder,
                tag.dtype,
                tag.count,
                0,
                planecount=uic2tag.count if uic2tag is not None else 1,
            )
        except Exception as exc:
            logger().warning(
                f'{self!r} <tifffile.read_uic1tag> raised {exc!r:.128}'
            )

    cdef void _process_special_format_tags_imagej(self):
        cdef TiffTag tag
        tag = self.tags.get(50839)
        if tag is not None:
            # decode IJMetadata tag
            try:
                tag.value = imagej_metadata(
                    tag.value,
                    self.tags.valueof(50838),  # IJMetadataByteCounts
                    self.tiff.byteorder_str,
                )
            except Exception as exc:
                logger().warning(
                    f'{self!r} <tifffile.imagej_metadata> raised {exc!r:.128}'
                )

    cdef void _process_special_format_tags_bitspersample_2(self):
        value = self.tags.valueof(258, None, 0)
        assert value is not None
        # LSM might list more items than samplesperpixel
        value = value[: self.samplesperpixel]
        if any(v - value[0] for v in value):
            self.bitspersample = value
        else:
            self.bitspersample = int(value[0])

    cdef void _process_special_format_tags_sampleformat_2(self):
        value = self.tags.valueof(339, None, 0)
        assert value is not None
        value = value[: self.samplesperpixel]
        if any(v - value[0] for v in value):
            self.sampleformat = value
        else:
            self.sampleformat = int(value[0])

    cdef void _process_special_format_tags_gdal(self):
        value = self.tags.valueof(42113)  # GDAL_NODATA
        if value is not None and self.dtype is not None:
            try:
                pytype = type(self.dtype.type(0).item())
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

    cdef int _process_data_pointers(self) noexcept nogil:
        """Process dataoffsets and databytecounts tags."""
        cdef bint success
        cdef int64_t i, n
        cdef int64_t first_offset, first_bytecount
        
        # Get data offsets - check multiple possible sources
        success = self.tags.valueof_int_array(self._dataoffsets, 324, 0)  # TileOffsets
        if not success:
            success = self.tags.valueof_int_array(self._dataoffsets, 273, 0)  # StripOffsets
            if not success:
                success = self.tags.valueof_int_array(self._dataoffsets, 513, 0)  # JPEGInterchangeFormat
                if not success:
                    #self._dataoffsets = empty_vec
                    #logger().error(f'{self!r} missing data offset tag')
                    return -1
        
        # Get data byte counts
        success = self.tags.valueof_int_array(self._databytecounts, 325, 0)  # TileByteCounts
        if not success:
            success = self.tags.valueof_int_array(self._databytecounts, 279, 0)  # StripByteCounts
            if not success:
                success = self.tags.valueof_int_array(self._databytecounts, 514, 0)  # JPEGInterchangeFormatLength
                #if not success:
                #    self._databytecounts = empty_vec

        # Special handling for NDPI files with JPEG McuStarts
        ''' # TODO NPDI
        cdef vector[int64_t] mcustarts
        cdef int64_t high
        if self.tags.contains_code(65426):
            success = self.tags.valueof_int_array(mcustarts, 65426, 0)
            if success and (self.tags.contains_code(65420) and self.tags.contains_code(271)): # is_ndpi
                high = self.tags.valueof_int(65432, -1, 0)
                if high != -1:
                    high <<= 32
                    for i in range(mcustarts.size()):
                        mcustarts[i] += high

                jpegheader = self.fh.read_at(self._dataoffsets[0], mcustarts[0])
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
                    # Create data pointers from MCU starts
                    i, n = mcustarts.size
                    first_offset = self._dataoffsets[0]
                    first_bytecount = self._databytecounts[0]
                    
                    # Calculate bytecounts as differences between consecutive offsets
                    # and calculate offsets
                    self._dataoffsets.clear()
                    self._databytecounts.clear()
                    self._databytecounts.reserve(n)
                    self._dataoffsets.reserve(n)
                    for i in range(n-1):
                        self._databytecounts.push_back(mcustarts[i+1] - mcustarts[i])
                    self._databytecounts.push_back(first_bytecount - mcustarts[n-1])

                    for i in range(n):
                        self._dataoffsets.push_back(mcustarts[i] + first_offset)
        '''

        # Fix incorrect number of strip bytecounts and offsets
        cdef int64_t maxstrips
        cdef size_t dataoffsets_size = self._dataoffsets.size()
        cdef size_t databytecounts_size = self._databytecounts.size()
        
        if self.imagelength != 0\
           and self.rowsperstrip\
           and not self.tags.contains_code(34412):
            maxstrips = (
                <int64_t>(
                    floor(self.imagelength + self.rowsperstrip - 1)
                    / self.rowsperstrip
                )
                * self.imagedepth
            )
            if self.planarconfig == 2:
                maxstrips *= self.samplesperpixel
                
            if maxstrips != databytecounts_size:
                return -1
                    
            if maxstrips != dataoffsets_size:
                return -1

        ## Create databytecounts if missing but required
        #if self._databytecounts.empty() and self.shape is not None and self.dtype is not None:
        #    self._databytecounts.push_back(product(self.shape) * (self.bitspersample // 8))
        #    if self.compression != 1:
        #        return -1
        # TODO: must be incorrect because self.shape is filled after
        if self._databytecounts.empty():
            return -1
        return 0

    cdef void _determine_shape_and_dtype(self):
        """Determine the shape and dtype of the image data."""
        # Determine dtype from sample format and bits per sample
        cdef str dtypestr = TIFF.SAMPLE_DTYPES.get(
            (self.sampleformat, self.bitspersample), None
        )
        if dtypestr is not None:
            dtype = numpy.dtype(dtypestr)
        else:
            dtype = None
        self.dtype = self._dtype = dtype

        # Determine shape based on dimensions and samples
        cdef int64_t imagelength = self.imagelength
        cdef int64_t imagewidth = self.imagewidth
        cdef int64_t imagedepth = self.imagedepth
        cdef int64_t samplesperpixel = self.samplesperpixel

        if self.photometric == 2 or samplesperpixel > 1:  # PHOTOMETRIC.RGB
            if self.planarconfig == 1:  # samples stored contiguously
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
            else:  # samples stored separately (planar)
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
        else:  # single channel image
            self.shaped = (1, imagedepth, imagelength, imagewidth, 1)
            if imagedepth == 1:
                self.shape = (imagelength, imagewidth)
                self.axes = 'YX'
            else:
                self.shape = (imagedepth, imagelength, imagewidth)
                self.axes = 'ZYX'

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
        if self.hash() in CACHED_DECODERS:
            return CACHED_DECODERS[self.hash()]
        
        # Create the decoder using the factory method
        decoder = TiffDecoder.create(self)
        
        # Cache the decoder
        CACHED_DECODERS[self.hash()] = decoder
        
        return decoder

    def segments(
        self,
        *,
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
        global THREADPOOL
        cdef TiffPage keyframe = self.keyframe  # self or keyframe
        cdef int i, n
        
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
            
        # Use vectors directly to create segments more efficiently
        dataoffsets = self._dataoffsets
        databytecounts = self._databytecounts
        n = dataoffsets.size()
        segments = [(self.fh, dataoffsets[i], databytecounts[i], i) for i in range(n)]
        
        if maxworkers < 2:
            for segment in segments:
                yield process(segment)
        else:
            # reduce memory overhead by processing chunks of up to
            # buffersize of segments because ThreadPoolExecutor.map is not
            # collecting iterables lazily
            yield from THREADPOOL.map(process, segments)

    def asarray(
        self,
        *,
        out: OutputType = None,
        squeeze: bool = True,
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

        if self._dataoffsets.size() == 0:
            raise TiffFileError('missing data offset')

        cdef FileHandle fh = self.fh
        if maxworkers is None or maxworkers < 1:
            maxworkers = keyframe.maxworkers
        if buffersize is None:
            buffersize = -1

        # Check if we need to open the file
        closed = fh.closed
        if closed:
            warnings.warn(
                f'{self!r} reading array from closed file', UserWarning
            )
            fh.open()
            
        try:
            # Try different methods based on the image data organization
            if (isinstance(out, str) and out == 'memmap' and 
                keyframe.is_memmappable()):
                # Use memory mapping for memmappable data
                result = self._asarray_memmap()
            
            elif keyframe.is_contiguous():
                # Handle contiguous data
                result = self._asarray_contiguous(out)
                
            elif (keyframe.jpegheader is not None and 
                  keyframe is self and 
                  self.tags.contains_code(273) and  # striped
                  self.is_tiled() and  # but reported as tiled
                  self.imagewidth <= 65500 and 
                  self.imagelength <= 65500):
                # Handle NDPI JPEG data
                result = self._asarray_ndpi_jpeg(out)
                
            else:
                # Handle tiled/striped data
                result = self._asarray_tiled(out, maxworkers, buffersize)
                
            # Reshape result if needed
            result.shape = keyframe.shaped
            if squeeze:
                try:
                    result.shape = keyframe.shape
                except ValueError as exc:
                    logger().warning(
                        f'{self!r} <asarray> failed to reshape '
                        f'{result.shape} to {keyframe.shape}, raised {exc!r:.128}'
                    )
                    
            return result
            
        finally:
            if closed:
                fh.close()

    cdef object _asarray_memmap(self):
        """Return memory-mapped array directly from file.
        
        Directly memory map the contiguous array in the file without 
        reading the data into memory. This is the most efficient method
        when the data is contiguous and uncompressed.
        
        Returns:
            Memory-mapped NumPy array.
        """
        cdef TiffPage keyframe = self.keyframe
        return self.fh.memmap_array(
            keyframe.tiff.byteorder + keyframe._dtype.char,
            keyframe.shaped,
            offset=self._dataoffsets[0],
        )

    cdef object _asarray_contiguous(self, out):
        """Return contiguous array data as NumPy array.
        
        Handles data that is stored contiguously in the file, either 
        uncompressed or with simple compression that can be directly decoded.
        
        Parameters:
            out: Output specification for the array.
            
        Returns:
            NumPy array of the image data.
        """
        cdef TiffPage keyframe = self.keyframe
        
        if keyframe.is_subsampled():
            raise NotImplementedError('chroma subsampling not supported')
            
        if out is not None:
            out = create_output(out, keyframe.shaped, keyframe._dtype)
            
        result = self.fh.read_array(
            keyframe.tiff.byteorder_str + keyframe._dtype.char,
            count=product(keyframe.shaped),
            offset=self._dataoffsets[0],
            out=out,
        )
        
        # Handle bit order if needed
        if keyframe.fillorder == 2:
            result = imagecodecs.bitorder_decode(result, out=result)
            
        # Handle prediction if needed
        if keyframe.predictor != 1:
            unpredict = UNPREDICTORS[keyframe.predictor]
            if keyframe.predictor == 1:
                result = unpredict(result, axis=-2, out=result)
            else:
                # floatpred cannot decode in-place
                out = unpredict(result, axis=-2, out=result)
                result[:] = out
                
        return result

    cdef object _asarray_ndpi_jpeg(self, out):
        """Return NDPI JPEG data as NumPy array.
        
        Handle the special case of NDPI JPEG data that is reported as tiled
        but stored as a single JPEG strip.
        
        Parameters:
            out: Output specification for the array.
            
        Returns:
            NumPy array of the decoded JPEG image.
        """
        # Read the entire JPEG strip
        data = self.fh.read_at(
            self.tags.valueof(273)[0],  # StripOffsets
            self.tags.valueof(279)[0]   # StripByteCounts
        )
        
        # Decompress the JPEG data
        decompress = DECOMPRESSORS[self.compression]
        result = decompress(
            data,
            bitspersample=self.bitspersample,
            out=out,
        )
        return result

    cdef object _asarray_tiled(self, out, int64_t maxworkers, int64_t buffersize):
        """Return tiled/striped image data as NumPy array.
        
        Process individual tiles or strips, potentially in parallel,
        and assemble them into a complete image.
        
        Parameters:
            out: Output specification for the array.
            maxworkers: Maximum number of worker threads.
            buffersize: Size of buffer for reading data.
            
        Returns:
            NumPy array of the assembled image data.
        """
        cdef TiffPage keyframe = self.keyframe
        
        # Initialize the TiffPage.decode function
        keyframe.decode  
        
        # Create output array
        result = create_output(out, keyframe.shaped, keyframe._dtype)
        
        # Define function to copy decoded segments to output array
        def func(
            decoderesult: tuple[
                NDArray[Any] | None,
                tuple[int, int, int, int, int],
                tuple[int, int, int, int],
            ],
            keyframe: TiffPage = keyframe,
            out: NDArray[Any] = result,
        ) -> None:
            """Copy decoded segment to output array.
            
            Parameters:
                decoderesult: Tuple containing decoded segment data and position info.
                keyframe: The TiffPage being processed.
                out: Output array to copy data into.
            """
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

        # Process all segments
        for _ in self.segments(
            func=func,
            maxworkers=maxworkers,
            buffersize=buffersize,
            sort=True,
            _fullsize=False,
        ):
            pass
            
        return result

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
        tagno = struct.unpack(
            tiff.tagnoformat, 
            fh.read_at(self.offset, tiff.tagnosize)
        )[0]
        offset_pos = self.offset + tiff.tagnosize + tagno * tiff.tagsize
        return int(
            struct.unpack(
                tiff.offsetformat, 
                fh.read_at(offset_pos, tiff.offsetsize)
            )[0]
        )

    def aspage(self) -> TiffPage:
        """Return TiffPage instance."""
        return self

    @property
    def keyframe(self) -> TiffPage:
        """Self."""
        return self

    @keyframe.setter
    def keyframe(self, index: TiffPage) -> None:
        return

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
    def dataoffsets(self):
        """Offsets to image data."""
        return tuple(self._dataoffsets)

    @property
    def databytecounts(self):
        """Byte counts of image data."""
        return tuple(self._databytecounts)

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

    cpdef int64_t hash(self):
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
        cdef vector[int64_t] offsets
        self.tags.valueof_int_array(offsets, 330, 0) # Is it ok ?
        result = TiffPages.from_parent(
            self.fh,
            self.tiff,
            offsets)
        self._cache["pages"] = result
        return result

    @property
    def maxworkers(self) -> int:
        """Maximum number of threads for decoding segments.

        A value of 0 disables multi-threading also when stacking pages.

        """
        if self.is_contiguous() or self.dtype is None:
            return 0
        if self.compression in \
            {
            6,  # jpeg
            7,  # jpeg
            22610,  # jpegxr
            33003,  # jpeg2k
            33004,  # jpeg2k
            33005,  # jpeg2k
            33007,  # alt_jpeg
            34712,  # jpeg2k
            34892,  # jpeg
            34933,  # png
            34934,  # jpegxr ZIF
            48124,  # jetraw
            50001,  # webp
            50002,  # jpegxl
            52546,  # jpegxl DNG
            }: # TIFF.IMAGE_COMPRESSIONS
            return min(MAXWORKERS, self._dataoffsets.size())
        bytecount = product(self.chunks) * self.dtype.itemsize
        if bytecount < 2048:
            # disable multi-threading for small segments
            return 0
        if self.compression == 5 and bytecount < 14336:
            # disable multi-threading for small LZW compressed segments
            return 0
        if self._dataoffsets.size() < 4:
            return 1
        if self.compression != 1 or self.fillorder != 1 or self.predictor != 1:
            if imagecodecs is not None:
                return min(MAXWORKERS, self._dataoffsets.size())
        return 2  # optimum for large number of uncompressed tiles

    cpdef bint is_contiguous(self):
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

        if self._dataoffsets.empty():
            self._cache["is_contiguous"] = False
            return False
        if self._dataoffsets.size() == 1:
            self._cache["is_contiguous"] = True
            return True
        if self.is_stk() or self.is_lsm():
            self._cache["is_contiguous"] = False
            return True
        if sum(self._databytecounts) != self.nbytes:
            self._cache["is_contiguous"] = False
            return False
        for i in range(self._dataoffsets.size() - 1):
            if self._databytecounts[i] == 0 or self._dataoffsets[i] + self._databytecounts[i] != self._dataoffsets[i + 1]:
                self._cache["is_contiguous"] = False
                return False
        self._cache["is_contiguous"] = True
        return True

    cpdef bint is_final(self):
        """Image data are stored in final form. Excludes byte-swapping."""
        return (
            self.is_contiguous()
            and self.fillorder == 1
            and self.predictor == 1
            and not self.is_subsampled()
        )

    cpdef bint is_memmappable(self):
        """Image data in file can be memory-mapped to NumPy array."""
        return (
            self.fh.is_file
            and self.is_final()
            # and (self.bitspersample == 8 or self.parent.isnative)
            # aligned?
            and self.dtype is not None
            and self._dataoffsets[0] % self.dtype.itemsize == 0
        )

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
            or self._dataoffsets.size() < 1
            or self._dataoffsets[0] == 0
            or self._databytecounts.empty()
            or self._databytecounts[0] < 11
        ):
            return False
        data = self.fh.read_at(self._dataoffsets[0] + 6, 4)
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