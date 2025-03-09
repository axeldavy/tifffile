
#cython: language_level=3
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

from libc.stdint cimport int32_t, int64_t
from .files cimport FileHandle
from .format cimport TiffFormat, ByteOrder
from .names cimport get_tag_names
from .types cimport DATATYPE

from .utils import pformat, stripnull, bytes2str, julian_datetime,\
    logger, astype, recarray2dict
from .types import TiffFileError, TIFF
#from .tifffile import *

from datetime import datetime as DateTime
import json
import math
import numpy
import os
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

'''
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
'''


def read_exif_ifd(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read EXIF tags from file."""
    exif = read_tags(fh, byteorder, offsetsize, TIFF.EXIF_TAGS, maxifds=1)[0]
    for name in ('ExifVersion', 'FlashpixVersion'):
        try:
            exif[name] = bytes2str(exif[name])
        except Exception:
            pass
    if 'UserComment' in exif:
        idcode = exif['UserComment'][:8]
        try:
            if idcode == b'ASCII\x00\x00\x00':
                exif['UserComment'] = bytes2str(exif['UserComment'][8:])
            elif idcode == b'UNICODE\x00':
                exif['UserComment'] = exif['UserComment'][8:].decode('utf-16')
        except Exception:
            pass
    return exif

def read_tags(
    fh: FileHandle,
    /,
    byteorder: ByteOrder,
    offsetsize: int,
    tagnames: TiffTagRegistry,
    *,
    maxifds: int | None = None,
    customtags: (
        dict[int, Callable[[FileHandle, ByteOrder, int, int, int], Any]] | None
    ) = None,
) -> list[dict[str, Any]]:
    """Read tag values from chain of IFDs.

    Parameters:
        fh:
            Binary file handle to read from.
            The file handle position must be at a valid IFD header.
        byteorder:
            Byte order of TIFF file.
        offsetsize:
            Size of offsets in TIFF file (8 for BigTIFF, else 4).
        tagnames:
            Map of tag codes to names.
            For example, :py:class:`_TIFF.GPS_TAGS` or
            :py:class:`_TIFF.IOP_TAGS`.
        maxifds:
            Maximum number of IFDs to read.
            By default, read the whole IFD chain.
        customtags:
            Mapping of tag codes to functions reading special tag value from
            file.

    Raises:
        TiffFileError: Invalid TIFF structure.

    Notes:
        This implementation does not support 64-bit NDPI files.

    """
    code: int
    dtype: int
    count: int
    valuebytes: bytes
    valueoffset: int

    if offsetsize == 4:
        offsetformat = byteorder + 'I'
        tagnosize = 2
        tagnoformat = byteorder + 'H'
        tagsize = 12
        tagformat1 = byteorder + 'HH'
        tagformat2 = byteorder + 'I4s'
    elif offsetsize == 8:
        offsetformat = byteorder + 'Q'
        tagnosize = 8
        tagnoformat = byteorder + 'Q'
        tagsize = 20
        tagformat1 = byteorder + 'HH'
        tagformat2 = byteorder + 'Q8s'
    else:
        raise ValueError('invalid offset size')

    if customtags is None:
        customtags = {}
    if maxifds is None:
        maxifds = 2**32

    result: list[dict[str, Any]] = []
    unpack = struct.unpack
    offset = fh.tell()
    while len(result) < maxifds:
        # loop over IFDs
        try:
            tagno = unpack(tagnoformat, fh.read(tagnosize))[0]
            if tagno > 4096:
                raise TiffFileError(f'suspicious number of tags {tagno}')
        except Exception as exc:
            logger().error(
                f'<tifffile.read_tags> corrupted tag list @{offset} '
                f'raised {exc!r:.128}'
            )
            break

        tags = {}
        data = fh.read(tagsize * tagno)
        pos = fh.tell()
        index = 0

        for _ in range(tagno):
            code, dtype = unpack(tagformat1, data[index : index + 4])
            count, valuebytes = unpack(
                tagformat2, data[index + 4 : index + tagsize]
            )
            index += tagsize
            name = tagnames.get(code, str(code))
            try:
                valueformat = TIFF.DATA_FORMATS[dtype]
            except KeyError:
                logger().error(f'invalid data type {dtype!r} for tag #{code}')
                continue

            valuesize = count * struct.calcsize(valueformat)
            if valuesize > offsetsize or code in customtags:
                valueoffset = unpack(offsetformat, valuebytes)[0]
                if valueoffset < 8 or valueoffset + valuesize > fh.size:
                    logger().error(
                        f'invalid value offset {valueoffset} for tag #{code}'
                    )
                    continue
                fh.seek(valueoffset)
                if code in customtags:
                    readfunc = customtags[code]
                    value = readfunc(fh, byteorder, dtype, count, offsetsize)
                elif dtype in {1, 2, 7}:
                    # BYTES, ASCII, UNDEFINED
                    value = fh.read(valuesize)
                    if len(value) != valuesize:
                        logger().warning(
                            '<tifffile.read_tags> '
                            f'could not read all values for tag #{code}'
                        )
                elif code in tagnames:
                    fmt = (
                        f'{byteorder}'
                        f'{count * int(valueformat[0])}'
                        f'{valueformat[1]}'
                    )
                    value = unpack(fmt, fh.read(valuesize))
                else:
                    value = read_numpy(fh, byteorder, dtype, count, offsetsize)
            elif dtype in {1, 2, 7}:
                # BYTES, ASCII, UNDEFINED
                value = valuebytes[:valuesize]
            else:
                fmt = (
                    f'{byteorder}'
                    f'{count * int(valueformat[0])}'
                    f'{valueformat[1]}'
                )
                value = unpack(fmt, valuebytes[:valuesize])

            process = (
                code not in customtags
                and code not in TIFF.TAG_TUPLE
                and dtype != 7  # UNDEFINED
            )
            if process and dtype == 2:
                # TIFF ASCII fields can contain multiple strings,
                #   each terminated with a NUL
                try:
                    value = bytes2str(stripnull(value, first=False).strip())
                except UnicodeDecodeError as exc:
                    logger().warning(
                        '<tifffile.read_tags> coercing invalid ASCII to bytes '
                        f'for tag #{code}, due to {exc!r:.128}'
                    )
            else:
                if code in TIFF.TAG_ENUM:
                    t = TIFF.TAG_ENUM[code]
                    try:
                        value = tuple(t(v) for v in value)
                    except ValueError as exc:
                        if code not in {259, 317}:
                            # ignore compression/predictor
                            logger().warning(
                                f'<tifffile.read_tags> tag #{code} '
                                f'raised {exc!r:.128}'
                            )
                if process and len(value) == 1:
                    value = value[0]
            tags[name] = value

        result.append(tags)

        # read offset to next page
        fh.seek(pos)
        offset = unpack(offsetformat, fh.read(offsetsize))[0]
        if offset == 0:
            break
        if offset >= fh.size:
            logger().error(f'<tifffile.read_tags> invalid next page {offset=}')
            break
        fh.seek(offset)

    return result

def read_gps_ifd(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read GPS tags from file."""
    return read_tags(fh, byteorder, offsetsize, TIFF.GPS_TAGS, maxifds=1)[0]


def read_interoperability_ifd(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read Interoperability tags from file."""
    return read_tags(fh, byteorder, offsetsize, TIFF.IOP_TAGS, maxifds=1)[0]


def read_bytes(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> bytes:
    """Read tag data from file."""
    count *= numpy.dtype(
        'B' if dtype == 2 else byteorder + TIFF.DATA_FORMATS[dtype][-1]
    ).itemsize
    data = fh.read(count)
    if len(data) != count:
        logger().warning(
            '<tifffile.read_bytes> '
            f'failed to read {count} bytes, got {len(data)})'
        )
    return data


def read_utf8(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> str:
    """Read unicode tag value from file."""
    return fh.read(count).decode()


def read_numpy(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> NDArray[Any]:
    """Read NumPy array tag value from file."""
    return fh.read_array(
        'b' if dtype == 2 else byteorder + TIFF.DATA_FORMATS[dtype][-1], count
    )


def read_colormap(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> NDArray[Any]:
    """Read ColorMap or TransferFunction tag value from file."""
    cmap = fh.read_array(byteorder + TIFF.DATA_FORMATS[dtype][-1], count)
    if count % 3 == 0:
        cmap.shape = (3, -1)
    return cmap


def read_json(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> Any:
    """Read JSON tag value from file."""
    data = fh.read(count)
    try:
        return json.loads(stripnull(data).decode())
    except ValueError as exc:
        logger().warning(f'<tifffile.read_json> raised {exc!r:.128}')
    return None


def read_mm_header(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read FluoView mm_header tag value from file."""
    meta = recarray2dict(
        fh.read_record(numpy.dtype(TIFF.MM_HEADER), shape=1, byteorder=byteorder)
    )
    meta['Dimensions'] = [
        (bytes2str(d[0]).strip(), d[1], d[2], d[3], bytes2str(d[4]).strip())
        for d in meta['Dimensions']
    ]
    d = meta['GrayChannel']
    meta['GrayChannel'] = (
        bytes2str(d[0]).strip(),
        d[1],
        d[2],
        d[3],
        bytes2str(d[4]).strip(),
    )
    return meta


def read_mm_stamp(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> NDArray[Any]:
    """Read FluoView mm_stamp tag value from file."""
    return fh.read_array(byteorder + 'f8', 8)


def read_uic1tag(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
    planecount: int = 0,
) -> dict[str, Any]:
    """Read MetaMorph STK UIC1Tag value from file.

    Return empty dictionary if planecount is unknown.

    """
    if dtype not in {4, 5} or byteorder != '<':
        raise ValueError(f'invalid UIC1Tag {byteorder}{dtype}')
    result = {}
    if dtype == 5:
        # pre MetaMorph 2.5 (not tested)
        values = fh.read_array('<u4', 2 * count).reshape(count, 2)
        result = {'ZDistance': values[:, 0] / values[:, 1]}
    else:
        for _ in range(count):
            tagid = struct.unpack('<I', fh.read(4))[0]
            if planecount > 1 and tagid in {28, 29, 37, 40, 41}:
                # silently skip unexpected tags
                fh.read(4)
                continue
            name, value = read_uic_tag(fh, tagid, planecount, True)
            if name == 'PlaneProperty':
                pos = fh.tell()
                fh.seek(value + 4)
                result.setdefault(name, []).append(read_uic_property(fh))
                fh.seek(pos)
            else:
                result[name] = value
    return result


def read_uic2tag(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, NDArray[Any]]:
    """Read MetaMorph STK UIC2Tag value from file."""
    if dtype != 5 or byteorder != '<':
        raise ValueError('invalid UIC2Tag')
    values = fh.read_array('<u4', 6 * count).reshape(count, 6)
    return {
        'ZDistance': values[:, 0] / values[:, 1],
        'DateCreated': values[:, 2],  # julian days
        'TimeCreated': values[:, 3],  # milliseconds
        'DateModified': values[:, 4],  # julian days
        'TimeModified': values[:, 5],  # milliseconds
    }


def read_uic3tag(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, NDArray[Any]]:
    """Read MetaMorph STK UIC3Tag value from file."""
    if dtype != 5 or byteorder != '<':
        raise ValueError('invalid UIC3Tag')
    values = fh.read_array('<u4', 2 * count).reshape(count, 2)
    return {'Wavelengths': values[:, 0] / values[:, 1]}


def read_uic4tag(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, NDArray[Any]]:
    """Read MetaMorph STK UIC4Tag value from file."""
    if dtype != 4 or byteorder != '<':
        raise ValueError('invalid UIC4Tag')
    result = {}
    while True:
        tagid: int = struct.unpack('<H', fh.read(2))[0]
        if tagid == 0:
            break
        name, value = read_uic_tag(fh, tagid, count, False)
        result[name] = value
    return result


def read_uic_tag(
    fh: FileHandle, tagid: int, planecount: int, offset: bool, /
) -> tuple[str, Any]:
    """Read single UIC tag value from file and return tag name and value.

    UIC1Tags use an offset.

    """

    def read_int() -> int:
        return int(struct.unpack('<I', fh.read(4))[0])

    def read_int2() -> tuple[int, int]:
        value = struct.unpack('<2I', fh.read(8))
        return int(value[0]), (value[1])

    try:
        name, dtype = TIFF.UIC_TAGS[tagid]
    except IndexError:
        # unknown tag
        return f'_TagId{tagid}', read_int()

    Fraction = TIFF.UIC_TAGS[4][1]

    if offset:
        pos = fh.tell()
        if dtype not in {int, None}:
            off = read_int()
            if off < 8:
                # undocumented cases, or invalid offset
                if dtype is str:
                    return name, ''
                if tagid == 41:  # AbsoluteZValid
                    return name, off
                logger().warning(
                    '<tifffile.read_uic_tag> '
                    f'invalid offset for tag {name!r} @{off}'
                )
                return name, off
            fh.seek(off)

    value: Any

    if dtype is None:
        # skip
        name = '_' + name
        value = read_int()
    elif dtype is int:
        # int
        value = read_int()
    elif dtype is Fraction:
        # fraction
        value = read_int2()
        value = value[0] / value[1]
    elif dtype is julian_datetime:
        # datetime
        value = read_int2()
        try:
            value = julian_datetime(*value)
        except Exception as exc:
            value = None
            logger().warning(
                f'<tifffile.read_uic_tag> reading {name} raised {exc!r:.128}'
            )
    elif dtype is read_uic_property:
        # ImagePropertyEx
        value = read_uic_property(fh)
    elif dtype is str:
        # pascal string
        size = read_int()
        if 0 <= size < 2**10:
            value = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
            value = bytes2str(stripnull(value))
        elif offset:
            value = ''
            logger().warning(
                f'<tifffile.read_uic_tag> invalid string in tag {name!r}'
            )
        else:
            raise ValueError(f'invalid string size {size}')
    elif planecount == 0:
        value = None
    elif dtype == '%ip':
        # sequence of pascal strings
        value = []
        for _ in range(planecount):
            size = read_int()
            if 0 <= size < 2**10:
                string = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
                string = bytes2str(stripnull(string))
                value.append(string)
            elif offset:
                logger().warning(
                    f'<tifffile.read_uic_tag> invalid string in tag {name!r}'
                )
            else:
                raise ValueError(f'invalid string size: {size}')
    else:
        # struct or numpy type
        dtype = '<' + dtype
        if '%i' in dtype:
            dtype = dtype % planecount
        if '(' in dtype:
            # numpy type
            value = fh.read_array(dtype, 1)[0]
            if value.shape[-1] == 2:
                # assume fractions
                value = value[..., 0] / value[..., 1]
        else:
            # struct format
            value = struct.unpack(dtype, fh.read(struct.calcsize(dtype)))
            if len(value) == 1:
                value = value[0]

    if offset:
        fh.seek(pos + 4)

    return name, value


def read_uic_property(fh: FileHandle, /) -> dict[str, Any]:
    """Read UIC ImagePropertyEx or PlaneProperty tag from file."""
    size = struct.unpack('B', fh.read(1))[0]
    name = bytes2str(struct.unpack(f'{size}s', fh.read(size))[0])
    flags, prop = struct.unpack('<IB', fh.read(5))
    if prop == 1:
        value = struct.unpack('II', fh.read(8))
        value = value[0] / value[1]
    else:
        size = struct.unpack('B', fh.read(1))[0]
        value = bytes2str(
            struct.unpack(f'{size}s', fh.read(size))[0]
        )  # type: ignore[assignment]
    return {'name': name, 'flags': flags, 'value': value}


def read_cz_lsminfo(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read CZ_LSMINFO tag value from file."""
    if byteorder != '<':
        raise ValueError('invalid CZ_LSMINFO structure')
    magic_number, structure_size = struct.unpack('<II', fh.read(8))
    if magic_number not in {50350412, 67127628}:
        raise ValueError('invalid CZ_LSMINFO structure')
    fh.seek(-8, os.SEEK_CUR)
    CZ_LSMINFO = TIFF.CZ_LSMINFO

    if structure_size < numpy.dtype(CZ_LSMINFO).itemsize:
        # adjust structure according to structure_size
        lsminfo: list[tuple[str, str]] = []
        size = 0
        for name, typestr in CZ_LSMINFO:
            size += numpy.dtype(typestr).itemsize
            if size > structure_size:
                break
            lsminfo.append((name, typestr))
    else:
        lsminfo = CZ_LSMINFO

    result = recarray2dict(
        fh.read_record(numpy.dtype(lsminfo), shape=1, byteorder=byteorder)
    )

    # read LSM info subrecords at offsets
    for name, reader in TIFF.CZ_LSMINFO_READERS.items():
        if reader is None:
            continue
        offset = result.get('Offset' + name, 0)
        if offset < 8:
            continue
        fh.seek(offset)
        try:
            result[name] = reader(fh)
        except ValueError:
            pass
    return result

def read_sis(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read OlympusSIS structure from file.

    No specification is available. Only few fields are known.

    """
    result: dict[str, Any] = {}

    (magic, minute, hour, day, month, year, name, tagcount) = struct.unpack(
        '<4s6xhhhhh6x32sh', fh.read(60)
    )

    if magic != b'SIS0':
        raise ValueError('invalid OlympusSIS structure')

    result['name'] = bytes2str(stripnull(name))
    try:
        result['datetime'] = DateTime(
            1900 + year, month + 1, day, hour, minute
        )
    except ValueError:
        pass

    data = fh.read(8 * tagcount)
    for i in range(0, tagcount * 8, 8):
        tagtype, count, offset = struct.unpack('<hhI', data[i : i + 8])
        fh.seek(offset)
        if tagtype == 1:
            # general data
            (lenexp, xcal, ycal, mag, camname, pictype) = struct.unpack(
                '<10xhdd8xd2x34s32s', fh.read(112)  # 220
            )
            m = math.pow(10, lenexp)
            result['pixelsizex'] = xcal * m
            result['pixelsizey'] = ycal * m
            result['magnification'] = mag
            result['cameraname'] = bytes2str(stripnull(camname))
            result['picturetype'] = bytes2str(stripnull(pictype))
        elif tagtype == 10:
            # channel data
            continue
            # TODO: does not seem to work?
            # (length, _, exptime, emv, _, camname, _, mictype,
            #  ) = struct.unpack('<h22sId4s32s48s32s', fh.read(152))  # 720
            # result['exposuretime'] = exptime
            # result['emvoltage'] = emv
            # result['cameraname2'] = bytes2str(stripnull(camname))
            # result['microscopename'] = bytes2str(stripnull(mictype))

    return result

def olympusini_metadata(inistr: str, /) -> dict[str, Any]:
    """Return OlympusSIS metadata from INI string.

    No specification is available.

    """

    def keyindex(key: str, /) -> tuple[str, int]:
        # split key into name and index
        index = 0
        i = len(key.rstrip('0123456789'))
        if i < len(key):
            index = int(key[i:]) - 1
            key = key[:i]
        return key, index

    result: dict[str, Any] = {}
    bands: list[dict[str, Any]] = []
    value: Any
    zpos: list[Any] | None = None
    tpos: list[Any] | None = None
    for line in inistr.splitlines():
        line = line.strip()
        if line == '' or line[0] == ';':
            continue
        if line[0] == '[' and line[-1] == ']':
            section_name = line[1:-1]
            result[section_name] = section = {}
            if section_name == 'Dimension':
                result['axes'] = axes = []
                result['shape'] = shape = []
            elif section_name == 'ASD':
                result[section_name] = []
            elif section_name == 'Z':
                if 'Dimension' in result:
                    result[section_name]['ZPos'] = zpos = []
            elif section_name == 'Time':
                if 'Dimension' in result:
                    result[section_name]['TimePos'] = tpos = []
            elif section_name == 'Band':
                nbands = result['Dimension']['Band']
                bands = [{'LUT': []} for _ in range(nbands)]
                result[section_name] = bands
                iband = 0
        else:
            key, value = line.split('=')
            if value.strip() == '':
                value = None
            elif ',' in value:
                value = tuple(astype(v) for v in value.split(','))
            else:
                value = astype(value)

            if section_name == 'Dimension':
                section[key] = value
                axes.append(key)
                shape.append(value)
            elif section_name == 'ASD':
                if key == 'Count':
                    result['ASD'] = [{}] * value
                else:
                    key, index = keyindex(key)
                    result['ASD'][index][key] = value
            elif section_name == 'Band':
                if key[:3] == 'LUT':
                    lut = bands[iband]['LUT']
                    value = struct.pack('<I', value)
                    lut.append(
                        [ord(value[0:1]), ord(value[1:2]), ord(value[2:3])]
                    )
                else:
                    key, iband = keyindex(key)
                    bands[iband][key] = value
            elif key[:4] == 'ZPos' and zpos is not None:
                zpos.append(value)
            elif key[:7] == 'TimePos' and tpos is not None:
                tpos.append(value)
            else:
                section[key] = value

    if 'axes' in result:
        sisaxes = {'Band': 'C'}
        axes = []
        shape = []
        for i, x in zip(result['shape'], result['axes']):
            if i > 1:
                axes.append(sisaxes.get(x, x[0].upper()))
                shape.append(i)
        result['axes'] = ''.join(axes)
        result['shape'] = tuple(shape)
    try:
        result['Z']['ZPos'] = numpy.array(
            result['Z']['ZPos'][: result['Dimension']['Z']], numpy.float64
        )
    except Exception:
        pass
    try:
        result['Time']['TimePos'] = numpy.array(
            result['Time']['TimePos'][: result['Dimension']['Time']],
            numpy.int32,
        )
    except Exception:
        pass
    for band in bands:
        band['LUT'] = numpy.array(band['LUT'], numpy.uint8)
    return result

def read_sis_ini(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read OlympusSIS INI string from file."""
    inistr = bytes2str(stripnull(fh.read(count)))
    try:
        return olympusini_metadata(inistr)
    except Exception as exc:
        logger().warning(f'<tifffile.olympusini_metadata> raised {exc!r:.128}')
        return {}


def read_tvips_header(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read TVIPS EM-MENU headers from file."""
    result: dict[str, Any] = {}
    header_v1 = TIFF.TVIPS_HEADER_V1
    header = fh.read_record(numpy.dtype(header_v1), shape=1, byteorder=byteorder)
    for name, typestr in header_v1:
        result[name] = header[name].tolist()
    if header['Version'] == 2:
        header_v2 = TIFF.TVIPS_HEADER_V2
        header = fh.read_record(numpy.dtype(header_v2), shape=1, byteorder=byteorder)
        if header['Magic'] != 0xAAAAAAAA:
            logger().warning(
                '<tifffile.read_tvips_header> invalid TVIPS v2 magic number'
            )
            return {}
        # decode utf16 strings
        for name, typestr in header_v2:
            if typestr.startswith('V'):
                s = header[name].tobytes().decode('utf-16', errors='ignore')
                result[name] = stripnull(s, null='\x00')
            else:
                result[name] = header[name].tolist()
        # convert nm to m
        for axis in 'XY':
            header['PhysicalPixelSize' + axis] /= 1e9
            header['PixelSize' + axis] /= 1e9
    elif header.version != 1:
        logger().warning(
            '<tifffile.read_tvips_header> unknown TVIPS header version'
        )
        return {}
    return result


def read_fei_metadata(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read FEI SFEG/HELIOS headers from file."""
    result: dict[str, Any] = {}
    section: dict[str, Any] = {}
    data = bytes2str(stripnull(fh.read(count)))
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


def read_cz_sem(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read Zeiss SEM tag from file.

    See https://sourceforge.net/p/gwyddion/mailman/message/29275000/ for
    unnamed values.

    """
    result: dict[str, Any] = {'': ()}
    value: Any
    key = None
    data = bytes2str(stripnull(fh.read(count)))
    for line in data.splitlines():
        if line.isupper():
            key = line.lower()
        elif key:
            try:
                name, value = line.split('=')
            except ValueError:
                try:
                    name, value = line.split(':', 1)
                except Exception:
                    continue
            value = value.strip()
            unit = ''
            try:
                v, u = value.split()
                number = astype(v, (int, float))
                if number != v:
                    value = number
                    unit = u
            except Exception:
                number = astype(value, (int, float))
                if number != value:
                    value = number
                if value in {'No', 'Off'}:
                    value = False
                elif value in {'Yes', 'On'}:
                    value = True
            result[key] = (name.strip(), value)
            if unit:
                result[key] += (unit,)
            key = None
        else:
            result[''] += (astype(line, (int, float)),)
    return result


def read_nih_image_header(
    fh: FileHandle,
    byteorder: ByteOrder,
    dtype: int,
    count: int,
    offsetsize: int,
    /,
) -> dict[str, Any]:
    """Read NIH_IMAGE_HEADER tag value from file."""
    arr = fh.read_record(TIFF.NIH_IMAGE_HEADER, shape=1, byteorder=byteorder)
    arr = arr.view(arr.dtype.newbyteorder(byteorder))
    result = recarray2dict(arr)
    result['XUnit'] = result['XUnit'][: result['XUnitSize']]
    result['UM'] = result['UM'][: result['UMsize']]
    return result



cdef object read_tag(int32_t tag,
                     bytes fh,
                     int64_t offset,
                     ByteOrder byteorder,
                     int64_t dtype,
                     int64_t count,
                     int64_t offsetsize):
    fh.seek(offset)
    if (tag == 301
        or tag == 320):
        return read_colormap(fh, byteorder, dtype, count, offsetsize)
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
        data = fh.read(count)
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
        return read_numpy(fh, byteorder, dtype, count, offsetsize)
    elif (tag == 34680
          or tag == 34682): # Helios NanoLab
        return read_fei_metadata(fh, byteorder, dtype, count, offsetsize)
    elif tag == 33628:
        return read_uic1tag(fh, byteorder, dtype, count, offsetsize)  # Universal Imaging Corp STK
    elif tag == 33629:
        return read_uic2tag(fh, byteorder, dtype, count, offsetsize)
    elif tag == 33630:
        return read_uic3tag(fh, byteorder, dtype, count, offsetsize)
    elif tag == 33631:
        return read_uic4tag(fh, byteorder, dtype, count, offsetsize)
    elif tag == 34118:
        return read_cz_sem(fh, byteorder, dtype, count, offsetsize)  # Carl Zeiss SEM
    elif tag == 34361:
        return read_mm_header(fh, byteorder, dtype, count, offsetsize)  # Olympus FluoView
    elif tag == 34362:
        return read_mm_stamp(fh, byteorder, dtype, count, offsetsize)
    elif tag == 34412:
        return read_cz_lsminfo(fh, byteorder, dtype, count, offsetsize)  # Carl Zeiss LSM
    elif tag == 37706:
        return read_tvips_header(fh, byteorder, dtype, count, offsetsize)  # TVIPS EMMENU
    elif tag == 43314:
        return read_nih_image_header(fh, byteorder, dtype, count, offsetsize)
            # 40001: read_bytes,
    elif tag == 51123:
        try:
            return json.loads(stripnull(fh.read(count)).decode())
        except ValueError as exc:
            logger().warning(f'<tifffile.read_json> raised {exc!r:.128}')
        return None
    elif tag == 33471:
        return read_sis_ini(fh, byteorder, dtype, count, offsetsize)
    elif tag == 33560:
        return read_sis(fh, byteorder, dtype, count, offsetsize)
    elif tag == 34665:
        return read_exif_ifd(fh, byteorder, dtype, count, offsetsize)
    elif tag == 34853:
        return read_gps_ifd(fh, byteorder, dtype, count, offsetsize)  # conflicts with OlympusSIS
    elif tag == 40965:
        return read_interoperability_ifd(fh, byteorder, dtype, count, offsetsize)
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

    def __init__(
        self,
        FileHandle fh,
        TiffFormat tiff_format,
        int64_t offset,
        int64_t code,
        int64_t datatype,
        int64_t count,
        object value,
        int64_t valueoffset
    ) -> None:
        self.fh = fh
        self.tiff_format = tiff_format
        self.offset = offset
        self.code = code
        self.count = count
        self._value = value
        self.valueoffset = valueoffset
        self.datatype = datatype

    @staticmethod
    def fromfile(
        FileHandle fh,
        TiffFormat tiff_format,
        object offset = None,
        bytes header = None,
        bint validate = True,
    ) -> TiffTag:
        """Return TiffTag instance from file.

        Parameters:
            fh:
                FileHandler instance tag is read from.
            tiff_format:
                TiffFormat instance that describes tag encoding
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

        cdef int64_t resolved_offset

        if offset is None:
            resolved_offset = fh.tell()
        else:
            resolved_offset = offset

        if header is None:
            fh.read_at(resolved_offset, tiff_format.tagsize)

        # Parse tag header
        cdef int64_t valueoffset = offset + tiff_format.tagsize - tiff_format.tagoffsetthreshold
        cdef int64_t code, datatype, count
        cdef object value, value_or_offset
        (code, datatype, count, value_or_offset) = tiff_format.parse_tag_header(header)

        cdef int64_t structsize
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
            return TiffTag(fh, tiff_format, offset, code, datatype, count, None, 0)

        cdef int64_t valuesize = count * structsize
        # value_or_offset is either an actual value to parse,
        # or an offset to the value in the file.
        # Case of offset
        if (valuesize > tiff_format.tagoffsetthreshold
            or readable_tag(code)):  # TODO: only works with offsets?
            valueoffset = tiff_format.interprete_offset(value_or_offset)
            if validate and no_delay_load(code):
                value = TiffTag._read_value(
                    fh, tiff_format, offset, code,
                    datatype, count, valueoffset
                )
            elif (valueoffset < 8
                  or valueoffset + valuesize > fh.size):
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
                    fh, tiff_format, offset, code,
                    datatype, count, valueoffset
                )
            else:
                value = None
        elif datatype in [1, 2, 7]:
            # direct value of type BYTES, ASCII, UNDEFINED
            value = value_or_offset[:valuesize]
        elif (tiff_format.is_ndpi
              and count == 1
              and datatype in [4, 9, 13]
              and value_or_offset[4:] != b'\x00\x00\x00\x00'):
            # NDPI IFD or LONG, for example, in StripOffsets or StripByteCounts
            value = struct.unpack('<Q', value_or_offset)
        else:
            valueformat = TIFF.DATA_FORMATS[datatype]
            fmt = (
                f'{tiff_format.byteorder}'
                f'{count * int(valueformat[0])}'
                f'{valueformat[1]}'
            )
            value = struct.unpack(fmt, value_or_offset[:valuesize])

        value = TiffTag._process_value(value, code, datatype, offset)
        cdef TiffTag tag = TiffTag.__new__(TiffTag)
        tag.fh = fh
        tag.tiff_format = tiff_format
        tag.offset = offset
        tag.code = code
        tag.datatype = datatype
        tag.count = count
        tag.value = value
        tag.valueoffset = valueoffset
        return tag

    @staticmethod
    cdef object _read_value(
        FileHandle fh,
        TiffFormat tiff_format,
        int64_t offset,
        int64_t code,
        int64_t datatype,
        int64_t count,
        int64_t valueoffset
    ):
        """Read tag value from file."""
        cdef int64_t structsize
        try:
            structsize = get_data_format_size(datatype)
        except KeyError as exc:
            raise TiffFileError(
                f'<tifffile.TiffTag {code} @{offset}> '
                f'invalid data type {datatype!r}'
            ) from exc

        cdef ByteOrder byteorder = tiff_format.byteorder
        cdef int64_t offsetsize = tiff_format.offsetsize
        cdef int64_t valuesize = count * structsize

        if valueoffset < 8 or valueoffset + valuesize > fh.size:
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
                    fh,
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
            value = fh.read_at(valueoffset, valuesize)
            if len(value) != valuesize:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> '
                    'could not read all values'
                )
        elif not(tag_is_tuple(code)) and count > 1024:
            fh.seek(valueoffset)
            value = read_numpy(fh, byteorder, datatype, count, offsetsize)
        else:
            valueformat = TIFF.DATA_FORMATS[datatype]
            value = struct.unpack(
                f'{byteorder}{count * int(valueformat[0])}{valueformat[1]}',
                fh.read_at(valueoffset, valuesize),
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
                    stripnull(<bytes>value, first=False).strip()
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
                value = tuple([t(v) for v in value])
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
                self.fh,
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

    def astuple(self):
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
        value,
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
            dtype = <DATATYPE>dtype#enumarg(DATATYPE, dtype)

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
                    value = str(value) # TODO enumstr
                else:
                    value = pformat(tuple(str(v) for v in value))
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

    cpdef void add(self, tag: TiffTag):
        """Add tag."""
        cdef int64_t code = tag.code
        cdef dict d
        for d in self._list:
            if code not in d:
                d[code] = tag
                break
        else:
            self._list.append({code: tag})

    cpdef list keys(self):
        """Return codes of all tags."""
        return list(self._dict.keys())

    @staticmethod
    cdef int _offset_key(self, TiffTag tag):
        return tag.offset

    cpdef list values(self):
        """Return all tags in order they are stored in file."""
        cdef list result = []
        for d in self._list:
            result.extend(d.values())
        return sorted(result, key=TiffTags._offset_key)

    @staticmethod
    cdef int _offset_key2(self, tuple element):
        return element[1].offset

    cpdef list items(self):
        """Return all (code, tag) pairs in order tags are stored in file."""
        cdef list result = []
        for d in self._list:
            result.extend(d.items())
        return sorted(result, key=TiffTags._offset_key2)

    cpdef object valueof(
        self,
        object key,
        object default = None,
        int64_t index = -1,
    ):
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
        cdef TiffTag tag = self.get(key, default=None, index=index)
        if tag is None:
            return default
        try:
            return tag.value
        except TiffFileError:
            return default  # corrupted tag

    cpdef TiffTag get(
        self,
        object key,
        TiffTag default = None,
        int64_t index = -1):
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
        if index == -1:
            if key in self._dict:
                if isinstance(key, int):
                    return self._dict[key]
                return self._dict[<int>key]
            if not isinstance(key, str):
                return default
            index = 0

        try:
            tags = self._list[index]
        except IndexError:
            return default

        if key in tags:
            if isinstance(key, int):
                return tags[key]
            return tags[<int>key]

        if not isinstance(key, str):
            return default

        for tag in tags.values():
            if tag.name == key:
                return tag

        return default

    cpdef object getall(
        self,
        object key,
        object default = None,
    ):
        """Return list of all tags by code or name if exists, else default.

        Parameters:
            key:
                Code or name of tags to return.
            default:
                Value to return if no tags are found.

        """
        cdef list result = []
        cdef dict tags
        cdef TiffTag tag

        for tags in self._list:
            if key in tags:
                if isinstance(key, int):
                    result.append(tags[key])
                else:
                    result.append(tags[<int>key])
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

    def __getitem__(self, object key):
        """Return first tag by code or name. Raise KeyError if not found."""
        cdef TiffTag tag

        if key in self._dict:
            if isinstance(key, int):
                return self._dict[key]
            return self._dict[<int>key]

        if not isinstance(key, str):
            raise KeyError(key)

        for tag in self._dict.values():
            if tag.name == key:
                return tag

        raise KeyError(key)

    def __setitem__(self, int64_t code, TiffTag tag):
        """Add tag."""
        assert tag.code == code
        self.add(tag)

    def __delitem__(self, key) -> None:
        """Delete all tags by code or name."""
        cdef bint found = False
        cdef dict tags
        cdef TiffTag tag

        for tags in self._list:
            if key in tags:
                found = True
                if isinstance(key, int):
                    del tags[key]
                else:
                    del tags[<int>key]
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

    def __contains__(self, object item):
        """Return if tag is in map."""
        cdef TiffTag tag
        
        if item in self._dict:
            return True
            
        if not isinstance(item, str):
            return False
            
        for tag in self._dict.values():
            if tag.name == item:
                return True
                
        return False

    def __iter__(self):
        """Return iterator over all tags."""
        return iter(self.values())

    def __len__(self):
        """Return number of tags."""
        cdef int size = 0
        cdef dict d
        
        for d in self._list:
            size += len(d)
            
        return size

    def __repr__(self) -> str:
        return f'<tifffile.TiffTags @0x{id(self):016X}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string with information about TiffTags."""
        cdef list info = []
        cdef list tlines = []
        cdef list vlines = []
        cdef TiffTag tag
        cdef str value
        cdef object tag_value

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

    def __init__(self, object arg):
        self._dict = {}
        self._list = [self._dict]
        self.update(arg)

    cpdef void update(self, object arg):
        """Add mapping of codes to names to registry.

        Parameters:
            arg: Mapping of codes to names.
        """
        cdef int64_t code
        cdef str name
        
        if isinstance(arg, TiffTagRegistry):
            self._list.extend(arg._list)
            return
            
        if isinstance(arg, dict):
            arg = list(arg.items())
            
        for code, name in arg:
            self.add(code, name)

    cpdef void add(self, int64_t code, str name):
        """Add code and name to registry."""
        cdef dict d
        
        for d in self._list:
            if code in d and d[code] == name:
                break
            if code not in d and name not in d:
                d[code] = name
                d[name] = code
                break
        else:
            self._list.append({code: name, name: code})

    @staticmethod
    cdef int _code_key(self, tuple element):
        return element[0]

    cpdef list items(self):
        """Return all registry items as (code, name)."""
        cdef list result = []
        cdef dict d
        cdef tuple i
        
        for d in self._list:
            for i in d.items():
                if isinstance(i[0], int):
                    result.append(i)
                    
        return sorted(result, key=TiffTagRegistry._code_key)

    cpdef object get(self, object key, object default=None):
        """Return first code or name if exists, else default.

        Parameters:
            key: tag code or name to lookup.
            default: value to return if key is not found.
        """
        cdef dict d
        
        for d in self._list:
            if key in d:
                return d[key]
                
        return default

    cpdef object getall(self, object key, object default=None):
        """Return list of all codes or names if exists, else default.

        Parameters:
            key: tag code or name to lookup.
            default: value to return if key is not found.
        """
        cdef list result = []
        cdef dict d
        
        for d in self._list:
            if key in d:
                result.append(d[key])
                
        return result if result else default

    def __getitem__(self, object key):
        """Return first code or name. Raise KeyError if not found."""
        cdef dict d
        
        for d in self._list:
            if key in d:
                return d[key]
                
        raise KeyError(key)

    def __delitem__(self, object key):
        """Delete all tags of code or name."""
        cdef bint found = False
        cdef dict d
        cdef object value
        
        for d in self._list:
            if key in d:
                found = True
                value = d[key]
                del d[key]
                del d[value]
                
        if not found:
            raise KeyError(key)

    def __contains__(self, object item):
        """Return if code or name is in registry."""
        cdef dict d
        
        for d in self._list:
            if item in d:
                return True
                
        return False

    def __iter__(self):
        """Return iterator over all items in registry."""
        return iter(self.items())

    def __len__(self):
        """Return number of registered tags."""
        cdef int size = 0
        cdef dict d
        
        for d in self._list:
            size += len(d)
            
        return size // 2

    def __repr__(self) -> str:
        return f'<tifffile.TiffTagRegistry @0x{id(self):016X}>'

    def __str__(self) -> str:
        return 'TiffTagRegistry(((\n  {}\n))'.format(
            ',\n  '.join(f'({code}, {name!r})' for code, name in self.items())
        )
