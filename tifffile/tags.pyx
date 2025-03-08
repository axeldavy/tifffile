
from libc.stdint cimport int32_t, int64_t
from .types cimport DATATYPE
from .format cimport TiffFormat
from .names cimport get_tag_names

from .tifffile import *

import struct
import warnings

cdef str get_numpy_dtype(ByteOrder byteorder,
                         int64_t datatype):
    """Return NumPy dtype string for given TIFF DATATYPE"""
    if byteorder == ByteOrder.II:
        if datatype == <int64_t>DATATYPE.BYTE:
            return '<1B'
        elif datatype == <int64_t>DATATYPE.ASCII:
            return 'b'#'<1s'
        elif datatype == <int64_t>DATATYPE.SHORT:
            return '<1H'
        elif datatype == <int64_t>DATATYPE.LONG:
            return '<1I'
        elif datatype == <int64_t>DATATYPE.RATIONAL:
            return '<2I'
        elif datatype == <int64_t>DATATYPE.SBYTE:
            return '<1b'
        elif datatype == <int64_t>DATATYPE.UNDEFINED:
            return '<1B'
        elif datatype == <int64_t>DATATYPE.SSHORT:
            return '<1h'
        elif datatype == <int64_t>DATATYPE.SLONG:
            return '<1i'
        elif datatype == <int64_t>DATATYPE.SRATIONAL:
            return '<2i'
        elif datatype == <int64_t>DATATYPE.FLOAT:
            return '<1f'
        elif datatype == <int64_t>DATATYPE.DOUBLE:
            return '<1d'
        elif datatype == <int64_t>DATATYPE.IFD:
            return '<1I'
        elif datatype == <int64_t>DATATYPE.LONG8:
            return '<1Q'
        elif datatype == <int64_t>DATATYPE.SLONG8:
            return '<1q'
        elif datatype == <int64_t>DATATYPE.IFD8:
            return '<1Q'
    elif byteorder == ByteOrder.MM:
        if datatype == <int64_t>DATATYPE.BYTE:
            return '>1B'
        elif datatype == <int64_t>DATATYPE.ASCII:
            return 'b'#'>1s'
        elif datatype == <int64_t>DATATYPE.SHORT:
            return '>1H'
        elif datatype == <int64_t>DATATYPE.LONG:
            return '>1I'
        elif datatype == <int64_t>DATATYPE.RATIONAL:
            return '>2I'
        elif datatype == <int64_t>DATATYPE.SBYTE:
            return '>1b'
        elif datatype == <int64_t>DATATYPE.UNDEFINED:
            return '>1B'
        elif datatype == <int64_t>DATATYPE.SSHORT:
            return '>1h'
        elif datatype == <int64_t>DATATYPE.SLONG:
            return '>1i'
        elif datatype == <int64_t>DATATYPE.SRATIONAL:
            return '>2i'
        elif datatype == <int64_t>DATATYPE.FLOAT:
            return '>1f'
        elif datatype == <int64_t>DATATYPE.DOUBLE:
            return '>1d'
        elif datatype == <int64_t>DATATYPE.IFD:
            return '>1I'
        elif datatype == <int64_t>DATATYPE.LONG8:
            return '>1Q'
        elif datatype == <int64_t>DATATYPE.SLONG8:
            return '>1q'
        elif datatype == <int64_t>DATATYPE.IFD8:
            return '>1Q'
    raise ValueError(f'Unknown tag type {datatype}')

cdef str get_numpy_dtype_single(ByteOrder byteorder,
                                int64_t datatype):
    """Return NumPy dtype string for given TIFF DATATYPE"""
    if byteorder == ByteOrder.II:
        if datatype == <int64_t>DATATYPE.BYTE:
            return '<B'
        elif datatype == <int64_t>DATATYPE.ASCII:
            return 'b'#'<s'
        elif datatype == <int64_t>DATATYPE.SHORT:
            return '<H'
        elif datatype == <int64_t>DATATYPE.LONG:
            return '<I'
        elif datatype == <int64_t>DATATYPE.RATIONAL:
            return '<I'
        elif datatype == <int64_t>DATATYPE.SBYTE:
            return '<b'
        elif datatype == <int64_t>DATATYPE.UNDEFINED:
            return '<B'
        elif datatype == <int64_t>DATATYPE.SSHORT:
            return '<h'
        elif datatype == <int64_t>DATATYPE.SLONG:
            return '<i'
        elif datatype == <int64_t>DATATYPE.SRATIONAL:
            return '<i'
        elif datatype == <int64_t>DATATYPE.FLOAT:
            return '<f'
        elif datatype == <int64_t>DATATYPE.DOUBLE:
            return '<d'
        elif datatype == <int64_t>DATATYPE.IFD:
            return '<I'
        elif datatype == <int64_t>DATATYPE.LONG8:
            return '<Q'
        elif datatype == <int64_t>DATATYPE.SLONG8:
            return '<q'
        elif datatype == <int64_t>DATATYPE.IFD8:
            return '<Q'
    elif byteorder == ByteOrder.MM:
        if datatype == <int64_t>DATATYPE.BYTE:
            return '>B'
        elif datatype == <int64_t>DATATYPE.ASCII:
            return 'b'#'>s'
        elif datatype == <int64_t>DATATYPE.SHORT:
            return '>H'
        elif datatype == <int64_t>DATATYPE.LONG:
            return '>I'
        elif datatype == <int64_t>DATATYPE.RATIONAL:
            return '>I'
        elif datatype == <int64_t>DATATYPE.SBYTE:
            return '>b'
        elif datatype == <int64_t>DATATYPE.UNDEFINED:
            return '>B'
        elif datatype == <int64_t>DATATYPE.SSHORT:
            return '>h'
        elif datatype == <int64_t>DATATYPE.SLONG:
            return '>i'
        elif datatype == <int64_t>DATATYPE.SRATIONAL:
            return '>i'
        elif datatype == <int64_t>DATATYPE.FLOAT:
            return '>f'
        elif datatype == <int64_t>DATATYPE.DOUBLE:
            return '>d'
        elif datatype == <int64_t>DATATYPE.IFD:
            return '>I'
        elif datatype == <int64_t>DATATYPE.LONG8:
            return '>Q'
        elif datatype == <int64_t>DATATYPE.SLONG8:
            return '>q'
        elif datatype == <int64_t>DATATYPE.IFD8:
            return '>Q'
    raise ValueError(f'Unknown tag type {datatype}')


cdef object read_numpy(bytes source,
                       int64_t offset,
                       ByteOrder byteorder,
                       int64_t dtype,
                       int64_t count,
                       int64_t offsetsize):
    """Read NumPy array tag value from file."""
    numpy_dtype = get_numpy_dtype_single(byteorder, dtype)

    cdef int nbytes = count * numpy_dtype.itemsize

    result = numpy.frombuffer(source[offset:offset+nbytes], dtype=numpy_dtype)

    if result.nbytes != nbytes:
        raise ValueError('size mismatch')

    if not result.dtype.isnative:
        if not numpy_dtype.isnative:
            result.byteswap(True)
        result = result.view(result.dtype.newbyteorder())
    elif result.dtype.isnative != numpy_dtype.isnative:
        result.byteswap(True)

    return result

cdef object read_tag(int32_t tag,
                     bytes source,
                     int64_t offset,
                     ByteOrder byteorder,
                     int64_t dtype,
                     int64_t count,
                     int64_t offsetsize):
    if (tag == 301
        or tag == 320):
        return read_colormap(source, offset, byteorder, dtype, count, offsetsize)
    elif (tag == 33723
          or tag == 37724
          or tag == 33923
          or tag == 40100
          or tag == 50288
          or tag == 50296
          or tag == 50839
          or tag == 65459):
        # read_bytes
        count *= get_data_format_size(dtype)
        data = source[offset:offset + count]
        if len(data) != count:
            logger().warning(
                '<tifffile.read_bytes> '
                f'failed to read {count} bytes, got {len(data)})'
            )
        return data
    elif (tag == 34363
          or tag == 34386
          or tag == 65426
          or tag == 65432
          or tag == 65439): # NDPI unknown
        return read_numpy(source, offset, byteorder, dtype, count, offsetsize)
    elif (tag == 34680
          or tag == 34682): # Helios NanoLab
        # read_fei_metadata
        result: dict[str, Any] = {}
        section: dict[str, Any] = {}
        data = bytes2str(stripnull(source[offset:offset + count]))
        for line in data.splitlines():
            line = line.strip()
            if line.startswith('['):
                section = {}
                result[line[1:-1]] = section
                continue
            try:
                key, value = line.split('=')
            except ValueError:
                continue
            section[key] = astype(value)  
        return result
    elif tag == 33628:
        return read_uic1tag(source, offset, byteorder, dtype, count, offsetsize)  # Universal Imaging Corp STK
    elif tag == 33629:
        return read_uic2tag(source, offset, byteorder, dtype, count, offsetsize)
    elif tag == 33630:
        return read_uic3tag(source, offset, byteorder, dtype, count, offsetsize)
    elif tag == 33631:
        return read_uic4tag(source, offset, byteorder, dtype, count, offsetsize)
    elif tag == 34118:
        return read_cz_sem(source, offset, byteorder, dtype, count, offsetsize)  # Carl Zeiss SEM
    elif tag == 34361:
        return read_mm_header(source, offset, byteorder, dtype, count, offsetsize)  # Olympus FluoView
    elif tag == 34362:
        return read_mm_stamp(source, offset, byteorder, dtype, count, offsetsize)
    elif tag == 34412:
        return read_cz_lsminfo(source, offset, byteorder, dtype, count, offsetsize)  # Carl Zeiss LSM
    elif tag == 37706:
        return read_tvips_header(source, offset, byteorder, dtype, count, offsetsize)  # TVIPS EMMENU
    elif tag == 43314:
        return read_nih_image_header(source, offset, byteorder, dtype, count, offsetsize)
            # 40001: read_bytes,
    elif tag == 51123:
        try:
            return json.loads(stripnull(source[offset:offset + count]).decode())
        except ValueError as exc:
            logger().warning(f'<tifffile.read_json> raised {exc!r:.128}')
        return None
    elif tag == 33471:
        return read_sis_ini(source, offset, byteorder, dtype, count, offsetsize)
    elif tag == 33560:
        return read_sis(source, offset, byteorder, dtype, count, offsetsize)
    elif tag == 34665:
        return read_exif_ifd(source, offset, byteorder, dtype, count, offsetsize)
    elif tag == 34853:
        return read_gps_ifd(source, offset, byteorder, dtype, count, offsetsize)  # conflicts with OlympusSIS
    elif tag == 40965:
        return read_interoperability_ifd(source, offset, byteorder, dtype, count, offsetsize)
    raise KeyError("Unknown tag code")

cdef bint readable_tag(int32_t tag):
    """Tags supported by read_tag"""
    if (tag == 301
        or tag == 320
        or tag == 33723
        or tag == 37724
        or tag == 33923
        or tag == 40100
        or tag == 50288
        or tag == 50296
        or tag == 50839
        or tag == 65459
        or tag == 34363
        or tag == 34386
        or tag == 65426
        or tag == 65432
        or tag == 65439
        or tag == 34680
        or tag == 34682
        or tag == 33628
        or tag == 33629
        or tag == 33630
        or tag == 33631
        or tag == 34118
        or tag == 34361
        or tag == 34362
        or tag == 34412
        or tag == 37706
        or tag == 43314
        or tag == 51123
        or tag == 33471
        or tag == 33560
        or tag == 34665
        or tag == 34853
        or tag == 40965):
        return True
    return False

cdef bint no_delay_load(int32_t tag):
    """tags whose values are not delay loaded"""
    if (tag == 258  # BitsPerSample
        or tag == 270  # ImageDescription
        or tag == 273  # StripOffsets
        or tag == 277  # SamplesPerPixel
        or tag == 279  # StripByteCounts
        or tag == 282  # XResolution
        or tag == 283  # YResolution
    #   or tag ==  301  # TransferFunction
        or tag == 305  # Software
    #   or tag == 306,  # DateTime
    #   or tag == 320,  # ColorMap
        or tag == 324  # TileOffsets
        or tag == 325  # TileByteCounts
        or tag == 330  # SubIFDs
        or tag == 338  # ExtraSamples
        or tag == 339  # SampleFormat
        or tag == 347  # JPEGTables
        or tag == 513 # JPEGInterchangeFormat
        or tag == 514  # JPEGInterchangeFormatLength
        or tag == 530  # YCbCrSubSampling
        or tag == 33628  # UIC1tag
        or tag == 42113  # GDAL_NODATA
        or tag == 50838  # IJMetadataByteCounts
        or tag == 50839):  # IJMetadata
        return True
    return False

cdef bint tag_is_tuple(int32_t tag):
    if (tag == 273
        or tag == 279
        or tag == 282
        or tag == 283
        or tag == 324
        or tag == 325
        or tag == 330
        or tag == 338
        or tag == 513
        or tag == 514
        or tag == 530
        or tag == 531
        or tag == 34736
        or tag == 50838):
        return True
    return False

cdef int64_t get_data_format_size(int64_t datatype) nogil:
    """Return size in bytes for the given TIFF DATATYPE.
    
    Parameters:
        datatype: TIFF DATATYPE value
        
    Returns:
        Size in bytes for the datatype
    """
    if datatype == <int64_t>DATATYPE.BYTE:
        return 1
    elif datatype == <int64_t>DATATYPE.ASCII:
        return 1
    elif datatype == <int64_t>DATATYPE.SHORT:
        return 2
    elif datatype == <int64_t>DATATYPE.LONG:
        return 4
    elif datatype == <int64_t>DATATYPE.RATIONAL:
        return 8  # 2 LONGs
    elif datatype == <int64_t>DATATYPE.SBYTE:
        return 1
    elif datatype == <int64_t>DATATYPE.UNDEFINED:
        return 1
    elif datatype == <int64_t>DATATYPE.SSHORT:
        return 2
    elif datatype == <int64_t>DATATYPE.SLONG:
        return 4
    elif datatype == <int64_t>DATATYPE.SRATIONAL:
        return 8  # 2 SLONGs
    elif datatype == <int64_t>DATATYPE.FLOAT:
        return 4
    elif datatype == <int64_t>DATATYPE.DOUBLE:
        return 8
    elif datatype == <int64_t>DATATYPE.IFD:
        return 4
    elif datatype == <int64_t>DATATYPE.LONG8:
        return 8
    elif datatype == <int64_t>DATATYPE.SLONG8:
        return 8
    elif datatype == <int64_t>DATATYPE.IFD8:
        return 8
    raise KeyError(f"Unknown TIFF DATATYPE {datatype}")

cdef object interprete_data_format(int64_t datatype, const void* data):
    """Interpret data according to TIFF DATATYPE and return as appropriate Python type.
    
    Parameters:
        datatype: TIFF DATATYPE value
        data: Pointer to binary data
        
    Returns:
        Tuple containing interpreted data
    """
    cdef unsigned char* bytes_data = <unsigned char*>data
    cdef unsigned short* short_data = <unsigned short*>data
    cdef unsigned int* int_data = <unsigned int*>data
    cdef unsigned long long* long_data = <unsigned long long*>data
    cdef char* char_data = <char*>data
    cdef short* sshort_data = <short*>data
    cdef int* sint_data = <int*>data
    cdef long long* slong_data = <long long*>data
    cdef float* float_data = <float*>data
    cdef double* double_data = <double*>data
    
    if datatype == <int64_t>DATATYPE.BYTE:
        return bytes_data[0]
    elif datatype == <int64_t>DATATYPE.ASCII:
        return bytes_data[0:1]  # Return as bytes
    elif datatype == <int64_t>DATATYPE.SHORT:
        return short_data[0]
    elif datatype == <int64_t>DATATYPE.LONG:
        return int_data[0]
    elif datatype == <int64_t>DATATYPE.RATIONAL:
        return (int_data[0], int_data[1])
    elif datatype == <int64_t>DATATYPE.SBYTE:
        return char_data[0]
    elif datatype == <int64_t>DATATYPE.UNDEFINED:
        return bytes_data[0]
    elif datatype == <int64_t>DATATYPE.SSHORT:
        return sshort_data[0]
    elif datatype == <int64_t>DATATYPE.SLONG:
        return sint_data[0]
    elif datatype == <int64_t>DATATYPE.SRATIONAL:
        return (sint_data[0], sint_data[1])
    elif datatype == <int64_t>DATATYPE.FLOAT:
        return float_data[0]
    elif datatype == <int64_t>DATATYPE.DOUBLE:
        return double_data[0]
    elif datatype == <int64_t>DATATYPE.IFD:
        return int_data[0]
    elif datatype == <int64_t>DATATYPE.LONG8:
        return long_data[0]
    elif datatype == <int64_t>DATATYPE.SLONG8:
        return slong_data[0]
    elif datatype == <int64_t>DATATYPE.IFD8:
        return long_data[0]
    return ()  # Return empty tuple for unknown datatypes

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

    cdef bytes source
    """TIFF file as memory-mapped bytes."""
    cdef TiffFormat tiff_format
    """TIFF format to interprete the tag belongs."""
    cdef int64_t offset
    """Position of tag structure in file."""
    cdef int64_t code
    """Decimal code of tag."""
    cdef int64_t datatype
    """:py:class:`DATATYPE` of tag value item."""
    cdef int64_t count
    """Number of items in tag value."""
    cdef int64_t valueoffset
    """Position of tag value in file."""
    cdef object _value

    def __init__(
        self,
        bytes source,
        TiffFormat tiff_format,
        int64_t offset,
        int64_t code,
        int64_t datatype,
        int64_t count,
        object value,
        int64_t valueoffset,
        /,
    ) -> None:
        self.source = source
        self.tiff_format = tiff_format
        self.offset = offset
        self.code = code
        self.count = count
        self._value = value
        self.valueoffset = valueoffset
        self.datatype = datatype

    @staticmethod
    def frombuffer(
        bytes source,
        TiffFormat tiff_format,
        int64_t offset,
        /,
        *,
        bytes header = None,
        bint validate = True,
    ) -> TiffTag:
        """Return TiffTag instance from file.

        Parameters:
            parent:
                TiffFile instance tag is read from.
            offset:
                Position of tag structure in file.
                The default is the position of the file handle.
            header:
                Tag structure as bytes.
                The default is read from the file.
            validate:
                Raise TiffFileError if data type or value offset are invalid.

        Raises:
            TiffFileError:
                Data type or value offset are invalid and `validate` is *True*.

        """

        if header is None:
            header = bytes[offset:offset+tiff_format.tagsize]

        cdef int64_t code, datatype, count
        cdef object value, value_or_offset
        valueoffset = offset + tiff_format.tagsize - tiff_format.tagoffsetthreshold
        (code, datatype, count, value_or_offset) = tiff_format.parse_tag_header(header)

        cdef int64_t structsize = get_data_format_size(datatype)
        '''
        try:
            structsize = get_data_format_size(datatype)
        except KeyError as exc:
            msg = (
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid data type {datatype!r}'
            )
            if validate:
                raise TiffFileError(msg) from exc
            logger().error(msg)
            return cls(parent, offset, code, datatype, count, None, 0)
        '''

        cdef int64_t valuesize = count * structsize
        # Case of value_or_offset -> offset
        if (valuesize > tiff_format.tagoffsetthreshold
            or readable_tag(code)):  # TODO: only works with offsets?
            valueoffset = tiff_format.interprete_offset(value_or_offset)
            if validate and no_delay_load(code):
                value = TiffTag._read_value(
                    source, tiff_format, offset, code,
                    datatype, count, valueoffset
                )
            elif (valueoffset < 8
                  or valueoffset + valuesize > len(source)):
                msg = (
                    f'<tifffile.TiffTag {code} @{offset}> '
                    f'invalid value offset {valueoffset}'
                )
                if validate:
                    raise TiffFileError(msg)
                logger().warning(msg)
                value = None
            elif no_delay_load(code):
                value = TiffTag._read_value(
                    source, tiff_format, offset, code,
                    datatype, count, valueoffset
                )
            else:
                value = None
        elif datatype in [1, 2, 7]:
            # BYTES, ASCII, UNDEFINED
            value = value_or_offset[:valuesize]
        elif (tiff_format.is_ndpi
              and count == 1
              and datatype in [4, 9, 13]
              and value_or_offset[4:] != b'\x00\x00\x00\x00'):
            # NDPI IFD or LONG, for example, in StripOffsets or StripByteCounts
            value = struct.unpack('<Q', value_or_offset)
        else:
            fmt = (
                f'{tiff_format.byteorder}'
                f'{count * int(valueformat[0])}'
                f'{valueformat[1]}'
            )
            value = struct.unpack(fmt, value_or_offset[:valuesize])

        value = TiffTag._process_value(value, code, datatype, offset)
        cdef TiffTag tag = TiffTag.__new__(TiffTag)
        tag.source = source
        tag.tiff_format = tiff_format
        tag.offset = tag
        tag.code = code
        tag.datatype = datatype
        tag.count = count
        tag.value = value
        tag.valueoffset = valueoffset
        return tag

    @staticmethod
    cdef object _read_value(
        bytes source,
        TiffFormat tiff_format,
        int64_t offset,
        int64_t code,
        int64_t datatype,
        int64_t count,
        int64_t valueoffset
    ):
        """Read tag value from file."""
        cdef int64_t structsize = get_data_format_size(datatype)
        '''
        try:
            structsize = get_data_format_size(datatype)
        except KeyError as exc:
            raise TiffFileError(
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid data type {datatype!r}'
            ) from exc
        '''

        cdef ByteOrder byteorder = tiff_format.byteorder
        cdef int64_t offsetsize = tiff_format.offsetsize
        cdef int64_t valuesize = count * structsize

        if valueoffset < 8 or valueoffset + valuesize > len(source):
            raise TiffFileError(
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid value offset {valueoffset}'
            )
        # if valueoffset % 2:
        #     logger().warning(
        #         f'<tifffile.TiffTag {code} @{offset}> '
        #         'value does not begin on word boundary'
        #     )

        if readable_tag(code):
            try:
                value = read_tag(
                    code,
                    source,
                    valueoffset,
                    byteorder,
                    datatype,
                    count,
                    offsetsize
                )
            except Exception as exc:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> raised {exc!r:.128}'
                )
            else:
                return value

        if datatype in [1, 2, 7]:
            # BYTES, ASCII, UNDEFINED
            value = bytes[valueoffset:(valueoffset+valuesize)]
            if len(value) != valuesize:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> '
                    'could not read all values'
                )
        elif not(tag_is_tuple(code)) and count > 1024:
            value = read_numpy(source, valueoffset, byteorder, datatype, count, offsetsize)
        else:
            value = struct.unpack(
                f'{byteorder}{count * int(valueformat[0])}{valueformat[1]}',
                source[valueoffset:(valueoffset+valuesize)],
            )
        return value

    @staticmethod
    cdef object _process_value(
        object value, int64_t code, int64_t datatype, int64_t offset
    ):
        """Process tag value."""
        if (value is None
            or datatype == <int64_t>DATATYPE.BYTE
            or datatype == <int64_t>DATATYPE.UNDEFINED
            or readable_tag(code)
            or not isinstance(value, (bytes, str, tuple))):
            return value

        if datatype == <int64_t>DATATYPE.ASCII:
            # TIFF ASCII fields can contain multiple strings,
            #   each terminated with a NUL
            try:
                value = bytes2str(
                    stripnull(cast(bytes, value), first=False).strip()
                )
            except UnicodeDecodeError as exc:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> '
                    f'coercing invalid ASCII to bytes, due to {exc!r:.128}'
                )
            return value

        if code in TIFF.TAG_ENUM:
            t = TIFF.TAG_ENUM[code]
            try:
                value = tuple(t(v) for v in value)
            except ValueError as exc:
                if code not in {259, 317}:  # ignore compression/predictor
                    logger().warning(
                        f'<tifffile.TiffTag {code} @{offset}> '
                        f'raised {exc!r:.128}'
                    )

        if len(value) == 1 and not(tag_is_tuple(code)):
            value = value[0]

        return value

    @property
    def value(self) -> object:
        """Value of tag, delay-loaded from file if necessary."""
        if self._value is None:
            # print(
            #     f'_read_value {self.code} {TIFF.TAGS.get(self.code)} '
            #     f'{self.datatype}[{self.count}] @{self.valueoffset} '
            # )
            value = TiffTag._read_value(
                self.source,
                self.tiff_format,
                self.offset,
                self.code,
                self.datatype,
                self.count,
                self.valueoffset,
            )
            self._value = TiffTag._process_value(
                value,
                self.code,
                self.datatype,
                self.offset,
            )
        return self._value

    @value.setter
    def value(self, object value) -> None:
        self._value = value

    @property
    def dtype_name(self) -> str:
        """Name of data type of tag value."""
        try:
            return self.dtype.name  # type: ignore[attr-defined]
        except AttributeError:
            return f'TYPE{self.dtype}'

    @property
    def name(self) -> str:
        """Name of tag from :py:attr:`_TIFF.TAGS` registry."""
        return TIFF.TAGS.get(self.code, str(self.code))

    @property
    def dataformat(self) -> str:
        """Data type as `struct.pack` format."""
        return TIFF.DATA_FORMATS[self.dtype]

    @property
    def valuebytecount(self) -> int:
        """Number of bytes of tag value in file."""
        return self.count * struct.calcsize(TIFF.DATA_FORMATS[self.dtype])

    def astuple(self) -> TagTuple:
        """Return tag code, dtype, count, and encoded value.

        The encoded value is read from file if necessary.

        """
        if isinstance(self.value, bytes):
            value = self.value
        else:
            tiff = self.parent.tiff
            dataformat = TIFF.DATA_FORMATS[self.dtype]
            count = self.count * int(dataformat[0])
            fmt = f'{tiff.byteorder}{count}{dataformat[1]}'
            try:
                if self.dtype == 2:
                    # ASCII
                    value = struct.pack(fmt, self.value.encode('ascii'))
                    if len(value) != count:
                        raise ValueError
                elif count == 1 and not isinstance(self.value, tuple):
                    value = struct.pack(fmt, self.value)
                else:
                    value = struct.pack(fmt, *self.value)
            except Exception as exc:
                if tiff.is_ndpi and count == 1:
                    raise ValueError(
                        'cannot pack 64-bit NDPI value to 32-bit dtype'
                    ) from exc
                fh = self.parent.filehandle
                pos = fh.tell()
                fh.seek(self.valueoffset)
                value = fh.read(struct.calcsize(fmt))
                fh.seek(pos)
        return self.code, int(self.dtype), self.count, value, True

    def overwrite(
        self,
        filehandle,
        value: Any,
        /,
        *,
        dtype: DATATYPE | int | str | None = None,
        erase: bool = True,
    ) -> TiffTag:
        """Write new tag value to file and return new TiffTag instance.

        Warning: changing tag values in TIFF files might result in corrupted
        files or have unexpected side effects.

        The packed value is appended to the file if it is longer than the
        old value. The file position is left where it was.

        Overwriting tag values in NDPI files > 4 GB is only supported if
        single integer values and new offsets do not exceed the 32-bit range.

        Parameters:
            value:
                New tag value to write.
                Must be compatible with the `struct.pack` formats corresponding
                to the tag's data type.
            dtype:
                New tag data type. By default, the data type is not changed.
            erase:
                Overwrite previous tag values in file with zeros.

        Raises:
            struct.error:
                Value is not compatible with dtype or new offset exceeds
                TIFF size limit.
            ValueError:
                Invalid value or dtype, or old integer value in NDPI files
                exceeds 32-bit range.

        """
        if self.offset < 8 or self.valueoffset < 8:
            raise ValueError(f'cannot rewrite tag at offset {self.offset} < 8')

        fh = filehandle
        tiff = self.tiff_format
        if tiff.is_ndpi:
            # only support files < 4GB
            if self.count == 1 and self.dtype in {4, 13}:
                if isinstance(self.value, tuple):
                    v = self.value[0]
                else:
                    v = self.value
                if v > 4294967295:
                    raise ValueError('cannot patch NDPI > 4 GB files')
            tiff = TIFF.CLASSIC_LE

        if value is None:
            value = b''
        if dtype is None:
            dtype = self.dtype
        elif isinstance(dtype, str):
            if len(dtype) > 1 and dtype[0] in '<>|=':
                dtype = dtype[1:]
            try:
                dtype = TIFF.DATA_DTYPES[dtype]
            except KeyError as exc:
                raise ValueError(f'unknown data type {dtype!r}') from exc
        else:
            dtype = enumarg(DATATYPE, dtype)

        packedvalue: bytes | None = None
        dataformat: str
        try:
            dataformat = TIFF.DATA_FORMATS[dtype]
        except KeyError as exc:
            raise ValueError(f'unknown data type {dtype!r}') from exc

        if dtype == 2:
            # strings
            if isinstance(value, str):
                # enforce 7-bit ASCII on Unicode strings
                try:
                    value = value.encode('ascii')
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        'TIFF strings must be 7-bit ASCII'
                    ) from exc
            elif not isinstance(value, bytes):
                raise ValueError('TIFF strings must be 7-bit ASCII')
            if len(value) == 0 or value[-1:] != b'\x00':
                value += b'\x00'
            count = len(value)
            value = (value,)

        elif isinstance(value, bytes):
            # pre-packed binary data
            dtsize = struct.calcsize(dataformat)
            if len(value) % dtsize:
                raise ValueError('invalid packed binary data')
            count = len(value) // dtsize
            packedvalue = value
            value = (value,)

        else:
            try:
                count = len(value)
            except TypeError:
                value = (value,)
                count = 1
            if dtype in {5, 10}:
                if count < 2 or count % 2:
                    raise ValueError('invalid RATIONAL value')
                count //= 2  # rational

        if packedvalue is None:
            packedvalue = struct.pack(
                f'{tiff.byteorder}{count * int(dataformat[0])}{dataformat[1]}',
                *value,
            )
        newsize = len(packedvalue)
        oldsize = self.count * struct.calcsize(TIFF.DATA_FORMATS[self.dtype])
        valueoffset = self.valueoffset

        pos = fh.tell()
        try:
            if dtype != self.dtype:
                # rewrite data type
                fh.seek(self.offset + 2)
                fh.write(struct.pack(tiff.byteorder + 'H', dtype))

            if oldsize <= tiff.tagoffsetthreshold:
                if newsize <= tiff.tagoffsetthreshold:
                    # inline -> inline: overwrite
                    fh.seek(self.offset + 4)
                    fh.write(struct.pack(tiff.tagformat2, count, packedvalue))
                else:
                    # inline -> separate: append to file
                    fh.seek(0, os.SEEK_END)
                    valueoffset = fh.tell()
                    if valueoffset % 2:
                        # value offset must begin on a word boundary
                        fh.write(b'\x00')
                        valueoffset += 1
                    # write new offset
                    fh.seek(self.offset + 4)
                    fh.write(
                        struct.pack(
                            tiff.tagformat2,
                            count,
                            struct.pack(tiff.offsetformat, valueoffset),
                        )
                    )
                    # write new value
                    fh.seek(valueoffset)
                    fh.write(packedvalue)

            elif newsize <= tiff.tagoffsetthreshold:
                # separate -> inline: erase old value
                valueoffset = (
                    self.offset + 4 + struct.calcsize(tiff.tagformat2[:2])
                )
                fh.seek(self.offset + 4)
                fh.write(struct.pack(tiff.tagformat2, count, packedvalue))
                if erase:
                    fh.seek(self.valueoffset)
                    fh.write(b'\x00' * oldsize)
            elif newsize <= oldsize or self.valueoffset + oldsize == fh.size:
                # separate -> separate smaller: overwrite, erase remaining
                fh.seek(self.offset + 4)
                fh.write(struct.pack(tiff.tagformat2[:2], count))
                fh.seek(self.valueoffset)
                fh.write(packedvalue)
                if erase and oldsize - newsize > 0:
                    fh.write(b'\x00' * (oldsize - newsize))
            else:
                # separate -> separate larger: erase old value, append to file
                fh.seek(0, os.SEEK_END)
                valueoffset = fh.tell()
                if valueoffset % 2:
                    # value offset must begin on a word boundary
                    fh.write(b'\x00')
                    valueoffset += 1
                # write offset
                fh.seek(self.offset + 4)
                fh.write(
                    struct.pack(
                        tiff.tagformat2,
                        count,
                        struct.pack(tiff.offsetformat, valueoffset),
                    )
                )
                # write value
                fh.seek(valueoffset)
                fh.write(packedvalue)
                if erase:
                    fh.seek(self.valueoffset)
                    fh.write(b'\x00' * oldsize)

        finally:
            fh.seek(pos)  # must restore file position

        return TiffTag(
            self.parent,
            self.offset,
            self.code,
            dtype,
            count,
            value,
            valueoffset,
        )

    def _fix_lsm_bitspersample(self) -> None:
        """Correct LSM bitspersample tag.

        Old LSM writers may use a separate region for two 16-bit values,
        although they fit into the tag value element of the tag.

        """
        if self.code != 258 or self.count != 2:
            return
        # TODO: test this case; need example file
        logger().warning(f'{self!r} correcting LSM bitspersample tag')
        value = struct.pack('<HH', *self.value)
        self.valueoffset = struct.unpack('<I', value)[0]
        self.parent.filehandle.seek(self.valueoffset)
        self.value = struct.unpack('<HH', self.parent.filehandle.read(4))

    def __repr__(self) -> str:
        name = '|'.join(TIFF.TAGS.getall(self.code, []))
        if name:
            name = ' ' + name
        return f'<tifffile.TiffTag {self.code}{name} @{self.offset}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string containing information about TiffTag."""
        height = 1 if detail <= 0 else 8 * detail
        dtype = self.dtype_name
        if self.count > 1:
            dtype += f'[{self.count}]'
        name = '|'.join(TIFF.TAGS.getall(self.code, []))
        if name:
            name = f'{self.code} {name} @{self.offset}'
        else:
            name = f'{self.code} @{self.offset}'
        line = f'TiffTag {name} {dtype} @{self.valueoffset} '
        line = line[:width]
        try:
            value = self.value
        except TiffFileError:
            value = 'CORRUPTED'
        else:
            try:
                if self.count == 1:
                    value = enumstr(value)
                else:
                    value = pformat(tuple(enumstr(v) for v in value))
            except Exception:
                if not isinstance(value, (tuple, list)):
                    pass
                elif height == 1:
                    value = value[:256]
                elif len(value) > 2048:
                    value = (
                        value[:1024] + value[-1024:]  # type: ignore[operator]
                    )
                value = pformat(value, width=width, height=height)
        if detail <= 0:
            line += '= '
            line += value[:width]
            line = line[:width]
        else:
            line += '\n' + value
        return line


cdef class TiffTags:
    """Multidict-like interface to TiffTag instances in TiffPage.

    Differences to a regular dict:

    - values are instances of :py:class:`TiffTag`.
    - keys are :py:attr:`TiffTag.code` (int).
    - multiple values can be stored per key.
    - can be indexed by :py:attr:`TiffTag.name` (`str`), slower than by key.
    - `iter()` returns values instead of keys.
    - `values()` and `items()` contain all values sorted by offset.
    - `len()` returns number of all values.
    - `get()` takes optional index argument.
    - some functions are not implemented, such as, `update` and `pop`.

    """

    cdef dict _dict#: dict[int, TiffTag]
    cdef list _list#: list[dict[int, TiffTag]]

    def __cinit__(self):
        self._dict = {}
        self._list = [self._dict]

    def add(self, tag: TiffTag, /) -> None:
        """Add tag."""
        code = tag.code
        for d in self._list:
            if code not in d:
                d[code] = tag
                break
        else:
            self._list.append({code: tag})

    def keys(self) -> list[int]:
        """Return codes of all tags."""
        return list(self._dict.keys())

    def values(self) -> list[TiffTag]:
        """Return all tags in order they are stored in file."""
        tags = (t for d in self._list for t in d.values())
        return sorted(tags, key=lambda t: t.offset)

    def items(self) -> list[tuple[int, TiffTag]]:
        """Return all (code, tag) pairs in order tags are stored in file."""
        items = (i for d in self._list for i in d.items())
        return sorted(items, key=lambda i: i[1].offset)

    def valueof(
        self,
        key: int | str,
        /,
        object default = None,
        index: int | None = None,
    ) -> Any:
        """Return value of tag by code or name if exists, else default.

        Parameters:
            key:
                Code or name of tag to return.
            default:
                Another value to return if specified tag is corrupted or
                not found.
            index:
                Specifies tag in case of multiple tags with identical code.
                The default is the first tag.

        """
        tag = self.get(key, default=None, index=index)
        if tag is None:
            return default
        try:
            return tag.value
        except TiffFileError:
            return default  # corrupted tag

    def get(
        self,
        key: int | str,
        /,
        TiffTag default = None,
        index: int | None = None,
    ) -> TiffTag | None:
        """Return tag by code or name if exists, else default.

        Parameters:
            key:
                Code or name of tag to return.
            default:
                Another tag to return if specified tag is corrupted or
                not found.
            index:
                Specifies tag in case of multiple tags with identical code.
                The default is the first tag.

        """
        if index is None:
            if key in self._dict:
                return self._dict[cast(int, key)]
            if not isinstance(key, str):
                return default
            index = 0
        try:
            tags = self._list[index]
        except IndexError:
            return default
        if key in tags:
            return tags[cast(int, key)]
        if not isinstance(key, str):
            return default
        for tag in tags.values():
            if tag.name == key:
                return tag
        return default

    def getall(
        self,
        key: int | str,
        /,
        object default = None,
    ) -> list[TiffTag] | None:
        """Return list of all tags by code or name if exists, else default.

        Parameters:
            key:
                Code or name of tags to return.
            default:
                Value to return if no tags are found.

        """
        result: list[TiffTag] = []
        for tags in self._list:
            if key in tags:
                result.append(tags[cast(int, key)])
            else:
                break
        if result:
            return result
        if not isinstance(key, str):
            return default
        for tags in self._list:
            for tag in tags.values():
                if tag.name == key:
                    result.append(tag)
                    break
            if not result:
                break
        return result if result else default

    def __getitem__(self, key: int | str, /) -> TiffTag:
        """Return first tag by code or name. Raise KeyError if not found."""
        if key in self._dict:
            return self._dict[cast(int, key)]
        if not isinstance(key, str):
            raise KeyError(key)
        for tag in self._dict.values():
            if tag.name == key:
                return tag
        raise KeyError(key)

    def __setitem__(self, code: int, tag: TiffTag, /) -> None:
        """Add tag."""
        assert tag.code == code
        self.add(tag)

    def __delitem__(self, key: int | str, /) -> None:
        """Delete all tags by code or name."""
        found = False
        for tags in self._list:
            if key in tags:
                found = True
                del tags[cast(int, key)]
            else:
                break
        if found:
            return
        if not isinstance(key, str):
            raise KeyError(key)
        for tags in self._list:
            for tag in tags.values():
                if tag.name == key:
                    del tags[tag.code]
                    found = True
                    break
            else:
                break
        if not found:
            raise KeyError(key)
        return

    def __contains__(self, item: object, /) -> bool:
        """Return if tag is in map."""
        if item in self._dict:
            return True
        if not isinstance(item, str):
            return False
        for tag in self._dict.values():
            if tag.name == item:
                return True
        return False

    def __iter__(self) -> Iterator[TiffTag]:
        """Return iterator over all tags."""
        return iter(self.values())

    def __len__(self) -> int:
        """Return number of tags."""
        size = 0
        for d in self._list:
            size += len(d)
        return size

    def __repr__(self) -> str:
        return f'<tifffile.TiffTags @0x{id(self):016X}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string with information about TiffTags."""
        info = []
        tlines = []
        vlines = []
        for tag in self:
            value = tag._str(width=width + 1)
            tlines.append(value[:width].strip())
            if detail > 0 and len(value) > width:
                try:
                    value = tag.value
                except Exception:
                    # delay load failed or closed file
                    continue
                if tag.code in {273, 279, 324, 325}:
                    if detail < 1:
                        value = value[:256]
                    elif len(value) > 1024:
                        value = value[:512] + value[-512:]
                    value = pformat(value, width=width, height=detail * 3)
                else:
                    value = pformat(value, width=width, height=detail * 8)
                if tag.count > 1:
                    vlines.append(
                        f'{tag.name} {tag.dtype_name}[{tag.count}]\n{value}'
                    )
                else:
                    vlines.append(f'{tag.name}\n{value}')
        info.append('\n'.join(tlines))
        if detail > 0 and vlines:
            info.append('\n')
            info.append('\n\n'.join(vlines))
        return '\n'.join(info)


cdef class TiffTagRegistry:
    """Registry of TIFF tag codes and names.

    Map tag codes and names to names and codes respectively.
    One tag code may be registered with several names, for example, 34853 is
    used for GPSTag or OlympusSIS2.
    Different tag codes may be registered with the same name, for example,
    37387 and 41483 are both named FlashEnergy.

    Parameters:
        arg: Mapping of codes to names.

    Examples:
        >>> tags = TiffTagRegistry([(34853, 'GPSTag'), (34853, 'OlympusSIS2')])
        >>> tags.add(37387, 'FlashEnergy')
        >>> tags.add(41483, 'FlashEnergy')
        >>> tags['GPSTag']
        34853
        >>> tags[34853]
        'GPSTag'
        >>> tags.getall(34853)
        ['GPSTag', 'OlympusSIS2']
        >>> tags.getall('FlashEnergy')
        [37387, 41483]
        >>> len(tags)
        4

    """

    cdef dict _dict#: dict[int | str, str | int]
    cdef list _list#: list[dict[int | str, str | int]]

    def __init__(
        self,
        arg: TiffTagRegistry | dict[int, str] | Sequence[tuple[int, str]],
        /,
    ) -> None:
        self._dict = {}
        self._list = [self._dict]
        self.update(arg)

    def update(
        self,
        arg: TiffTagRegistry | dict[int, str] | Sequence[tuple[int, str]],
        /,
    ) -> None:
        """Add mapping of codes to names to registry.

        Parameters:
            arg: Mapping of codes to names.

        """
        if isinstance(arg, TiffTagRegistry):
            self._list.extend(arg._list)
            return
        if isinstance(arg, dict):
            arg = list(arg.items())
        for code, name in arg:
            self.add(code, name)

    def add(self, code: int, name: str, /) -> None:
        """Add code and name to registry."""
        for d in self._list:
            if code in d and d[code] == name:
                break
            if code not in d and name not in d:
                d[code] = name
                d[name] = code
                break
        else:
            self._list.append({code: name, name: code})

    def items(self) -> list[tuple[int, str]]:
        """Return all registry items as (code, name)."""
        items = (
            i for d in self._list for i in d.items() if isinstance(i[0], int)
        )
        return sorted(items, key=lambda i: i[0])  # type: ignore[arg-type]

    @overload
    def get(self, key: int, /, default: None) -> str | None: ...

    @overload
    def get(self, key: str, /, default: None) -> int | None: ...

    @overload
    def get(self, key: int, /, default: str) -> str: ...

    def get(
        self, key: int | str, /, default: str | None = None
    ) -> str | int | None:
        """Return first code or name if exists, else default.

        Parameters:
            key: tag code or name to lookup.
            default: value to return if key is not found.

        """
        for d in self._list:
            if key in d:
                return d[key]
        return default

    @overload
    def getall(self, key: int, /, default: None) -> list[str] | None: ...

    @overload
    def getall(self, key: str, /, default: None) -> list[int] | None: ...

    @overload
    def getall(self, key: int, /, default: list[str]) -> list[str]: ...

    def getall(
        self, key: int | str, /, default: list[str] | None = None
    ) -> list[str] | list[int] | None:
        """Return list of all codes or names if exists, else default.

        Parameters:
            key: tag code or name to lookup.
            default: value to return if key is not found.

        """
        result = [d[key] for d in self._list if key in d]
        return result if result else default  # type: ignore[return-value]

    @overload
    def __getitem__(self, key: int, /) -> str: ...

    @overload
    def __getitem__(self, key: str, /) -> int: ...

    def __getitem__(self, key: int | str, /) -> int | str:
        """Return first code or name. Raise KeyError if not found."""
        for d in self._list:
            if key in d:
                return d[key]
        raise KeyError(key)

    def __delitem__(self, key: int | str, /) -> None:
        """Delete all tags of code or name."""
        found = False
        for d in self._list:
            if key in d:
                found = True
                value = d[key]
                del d[key]
                del d[value]
        if not found:
            raise KeyError(key)

    def __contains__(self, item: int | str, /) -> bool:
        """Return if code or name is in registry."""
        for d in self._list:
            if item in d:
                return True
        return False

    def __iter__(self) -> Iterator[tuple[int, str]]:
        """Return iterator over all items in registry."""
        return iter(self.items())

    def __len__(self) -> int:
        """Return number of registered tags."""
        size = 0
        for d in self._list:
            size += len(d)
        return size // 2

    def __repr__(self) -> str:
        return f'<tifffile.TiffTagRegistry @0x{id(self):016X}>'

    def __str__(self) -> str:
        return 'TiffTagRegistry(((\n  {}\n))'.format(
            ',\n  '.join(f'({code}, {name!r})' for code, name in self.items())
        )
