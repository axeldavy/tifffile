#cython: language_level=3
#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

cimport cython
from cython.operator cimport dereference, postincrement
from libc.stdint cimport uint16_t, int32_t, uint32_t, int64_t, uint64_t
from libcpp.string cimport string as cpp_string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_multimap as cpp_multimap
from libcpp.utility cimport pair
from .files cimport FileHandle
from .format cimport TiffFormat, ByteOrder, TagHeader
from .names cimport get_tag_names
from .types cimport DATATYPE
from .utils cimport bytes2str_stripnull, bytes2str_stripnull_last

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

cdef dict read_exif_ifd(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read EXIF tags from file."""
    exif = read_tags(fh, byteorder, offsetsize, TIFF.EXIF_TAGS, 1)[0]
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

cdef list read_tags(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t offsetsize,
    TiffTagRegistry tagnames,
    int64_t maxifds
):
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
    cdef int64_t code
    cdef int64_t datatype
    cdef int64_t count
    cdef unsigned char* valuebytes
    cdef int64_t valueoffset
    cdef object customtags = None
    cdef uint16_t tagno
    cdef uint32_t offset_value_le
    cdef uint64_t offset_value_be
    cdef int64_t tagnosize, tagsize, index, valuesize
    cdef bint is_little_endian = byteorder == ByteOrder.II
    cdef bint process

    if offsetsize == 4:
        tagnosize = 2
        tagsize = 12
    elif offsetsize == 8:
        tagnosize = 8
        tagsize = 20
    else:
        raise ValueError('invalid offset size')

    if customtags is None:
        customtags = {}
    if maxifds is None:
        maxifds = 2**32

    cdef list result = []
    cdef int64_t offset = fh.tell()
    cdef int64_t pos

    cdef bytes data_b
    cdef unsigned char* data
    cdef dict tags
    cdef str name
    
    while len(result) < maxifds:
        # loop over IFDs
        try:
            # Read tagno based on endianness
            data_b = fh.read(tagnosize)
            data = data_b
            if is_little_endian:
                if tagnosize == 2:
                    tagno = (<uint16_t*>data)[0]
                else:  # tagnosize == 8
                    tagno = (<uint64_t*>data)[0]
            else:
                if tagnosize == 2:
                    tagno = _bswap16((<uint16_t*>data)[0])
                else:  # tagnosize == 8
                    tagno = _bswap64((<uint64_t*>data)[0])
                
            if tagno > 4096:
                raise TiffFileError(f'suspicious number of tags {tagno}')
        except Exception as exc:
            logger().error(
                f'<tifffile.read_tags> corrupted tag list @{offset} '
                f'raised {exc!r:.128}'
            )
            break

        tags = {}
        data_b = fh.read(tagsize * tagno)
        data = data_b
        pos = fh.tell()
        index = 0

        for _ in range(tagno):
            # Read code and datatype directly based on endianness
            if is_little_endian:
                code = (<uint16_t*>&data[index])[0]
                datatype = (<uint16_t*>&data[index + 2])[0]
                count = (<uint32_t*>&data[index + 4])[0] if offsetsize == 4 else (<uint64_t*>&data[index + 4])[0]
                valuebytes = &data[index + 8]#:index + tagsize]
            else:
                code = _bswap16((<uint16_t*>&data[index])[0])
                datatype = _bswap16((<uint16_t*>&data[index + 2])[0])
                count = _bswap32((<uint32_t*>&data[index + 4])[0]) if offsetsize == 4 else _bswap64((<uint64_t*>&data[index + 4])[0])
                valuebytes = &data[index + 8]#:index + tagsize]
            
            index += tagsize
            name = tagnames.get(code, str(code))
            
            try:
                valuesize = count * get_data_format_size(datatype)
            except KeyError:
                logger().error(f'invalid data type {datatype!r} for tag #{code}')
                continue

            if valuesize > offsetsize or code in customtags:
                # Get valueoffset based on endianness and offsetsize
                if is_little_endian:
                    if offsetsize == 4:
                        valueoffset = (<uint32_t*>valuebytes)[0]
                    else:  # offsetsize == 8
                        valueoffset = (<uint64_t*>valuebytes)[0]
                else:
                    if offsetsize == 4:
                        valueoffset = _bswap32((<uint32_t*>valuebytes)[0])
                    else:  # offsetsize == 8
                        valueoffset = _bswap64((<uint64_t*>valuebytes)[0])
                
                if valueoffset < 8 or valueoffset + valuesize > fh.size:
                    logger().error(
                        f'invalid value offset {valueoffset} for tag #{code}'
                    )
                    continue
                    
                fh.seek(valueoffset)
                if code in customtags:
                    readfunc = customtags[code]
                    value = readfunc(fh, byteorder, datatype, count, offsetsize)
                elif datatype in {DATATYPE.BYTE, DATATYPE.ASCII, DATATYPE.UNDEFINED}:
                    value = fh.read(valuesize)
                    if len(value) != valuesize:
                        logger().warning(
                            '<tifffile.read_tags> '
                            f'could not read all values for tag #{code}'
                        )
                elif code in tagnames:
                    # Use optimized reading based on datatype and byteorder
                    value = read_formatted_data(fh, byteorder, datatype, count, valuesize)
                else:
                    value = read_numpy(fh, byteorder, datatype, count, offsetsize)
            elif datatype in {DATATYPE.BYTE, DATATYPE.ASCII, DATATYPE.UNDEFINED}:
                value = valuebytes[:valuesize]
            else:
                # Process inline data based on datatype and byteorder
                value = process_inline_data(valuebytes, byteorder, datatype, count, valuesize)

            process = (
                code not in customtags
                and not tag_is_tuple(code)
                and datatype != DATATYPE.UNDEFINED
            )
            
            if process and datatype == DATATYPE.ASCII:
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
                        value = tuple([t(v) for v in value])
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

        # Read offset to next page based on endianness and offsetsize
        fh.seek(pos)
        data_b = fh.read(offsetsize)
        data = data_b
        if is_little_endian:
            if offsetsize == 4:
                offset = (<uint32_t*>data)[0]
            else:  # offsetsize == 8
                offset = (<uint64_t*>data)[0]
        else:
            if offsetsize == 4:
                offset = _bswap32((<uint32_t*>data)[0])
            else:  # offsetsize == 8
                offset = _bswap64((<uint64_t*>data)[0])
        
        if offset == 0:
            break
        if offset >= fh.size:
            logger().error(f'<tifffile.read_tags> invalid next page {offset=}')
            break
        fh.seek(offset)

    return result

cdef extern from * nogil:
    """
    uint16_t _bswap16(uint16_t value) {
        return ((value & 0xFF) << 8) | ((value >> 8) & 0xFF);
    }

    uint32_t _bswap32(uint32_t value) {
        return ((value & 0xFF) << 24) | ((value & 0xFF00) << 8) |
               ((value >> 8) & 0xFF00) | ((value >> 24) & 0xFF);
    }

    uint64_t _bswap64(uint64_t value) {
        return ((value & 0xFF) << 56) | ((value & 0xFF00) << 40) |
               ((value & 0xFF0000) << 24) | ((value & 0xFF000000) << 8) |
               ((value >> 8) & 0xFF000000) | ((value >> 24) & 0xFF0000) |
               ((value >> 40) & 0xFF00) | ((value >> 56) & 0xFF);
    }
    """
    cdef uint16_t _bswap16(uint16_t value)
    cdef uint32_t _bswap32(uint32_t value)
    cdef uint64_t _bswap64(uint64_t value)

cdef tuple read_formatted_data(FileHandle fh, ByteOrder byteorder, int64_t datatype, 
                              int64_t count, int64_t valuesize):
    """Read data from file with specified format without string concatenation."""
    cdef bytes data = fh.read(valuesize)
    return process_inline_data(data, byteorder, datatype, count, valuesize)

cdef tuple process_inline_data(bytes data_b, ByteOrder byteorder, int64_t datatype, 
                              int64_t count, int64_t valuesize):
    """Process inline data with specified format without string concatenation."""
    cdef bint is_little_endian = byteorder == ByteOrder.II
    cdef int64_t i
    cdef list result = []
    cdef unsigned char* data = data_b
    
    # Process data based on datatype and endianness
    if is_little_endian:
        if datatype == <int64_t>DATATYPE.SHORT:
            for i in range(0, valuesize, 2):
                result.append((<uint16_t*>&data[i])[0])
        elif datatype == <int64_t>DATATYPE.LONG:
            for i in range(0, valuesize, 4):
                result.append((<uint32_t*>&data[i])[0])
        elif datatype == <int64_t>DATATYPE.RATIONAL:
            for i in range(0, valuesize, 8):
                result.append(((<uint32_t*>&data[i])[0], (<uint32_t*>&data[i+4])[0]))
        elif datatype == <int64_t>DATATYPE.SSHORT:
            for i in range(0, valuesize, 2):
                result.append((<short*>&data[i])[0])
        elif datatype == <int64_t>DATATYPE.SLONG:
            for i in range(0, valuesize, 4):
                result.append((<int*>&data[i])[0])
        elif datatype == <int64_t>DATATYPE.SRATIONAL:
            for i in range(0, valuesize, 8):
                result.append(((<int*>&data[i])[0], (<int*>&data[i+4])[0]))
        elif datatype == <int64_t>DATATYPE.FLOAT:
            for i in range(0, valuesize, 4):
                result.append((<float*>&data[i])[0])
        elif datatype == <int64_t>DATATYPE.DOUBLE:
            for i in range(0, valuesize, 8):
                result.append((<double*>&data[i])[0])
        else:
            assert False, f'Unknown datatype {datatype}'
        # Add more datatypes as needed
    else:  # big endian
        if datatype == <int64_t>DATATYPE.SHORT:
            for i in range(0, valuesize, 2):
                result.append(_bswap16((<uint16_t*>&data[i])[0]))
        elif datatype == <int64_t>DATATYPE.LONG:
            for i in range(0, valuesize, 4):
                result.append(_bswap32((<uint32_t*>&data[i])[0]))
        elif datatype == <int64_t>DATATYPE.RATIONAL:
            for i in range(0, valuesize, 8):
                result.append((_bswap32((<uint32_t*>&data[i])[0]), 
                              _bswap32((<uint32_t*>&data[i+4])[0])))
        elif datatype == <int64_t>DATATYPE.SSHORT:
            for i in range(0, valuesize, 2):
                result.append(<short>_bswap16((<uint16_t*>&data[i])[0]))
        elif datatype == <int64_t>DATATYPE.SLONG:
            for i in range(0, valuesize, 4):
                result.append(<int>_bswap32((<uint32_t*>&data[i])[0]))
        elif datatype == <int64_t>DATATYPE.SRATIONAL:
            for i in range(0, valuesize, 8):
                result.append((<int>_bswap32((<uint32_t*>&data[i])[0]), 
                              <int>_bswap32((<uint32_t*>&data[i+4])[0])))
        # Add more datatypes as needed
        else:
            assert False, f'Unknown datatype {datatype}'
        
    return tuple(result)

cdef dict read_gps_ifd(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read GPS tags from file."""
    return read_tags(fh, byteorder, offsetsize, TIFF.GPS_TAGS, 1)[0]


cdef dict read_interoperability_ifd(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read Interoperability tags from file."""
    return read_tags(fh, byteorder, offsetsize, TIFF.IOP_TAGS, 1)[0]


cdef bytes read_bytes(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read tag data from file."""
    count *= get_data_format_size(datatype)
    data = fh.read(count)
    if len(data) != count:
        logger().warning(
            '<tifffile.read_bytes> '
            f'failed to read {count} bytes, got {len(data)})'
        )
    return data


cdef str read_utf8(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read unicode tag value from file."""
    return fh.read(count).decode()


cdef object read_numpy(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read NumPy array tag value from file."""
    return fh.read_array(
        get_numpy_dtype_single(byteorder, datatype), count
    )


cdef object read_colormap(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read ColorMap or TransferFunction tag value from file."""
    cmap = fh.read_array(get_numpy_dtype_single(byteorder, datatype), count)
    if count % 3 == 0:
        cmap.shape = (3, -1)
    return cmap


cdef object read_json(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read JSON tag value from file."""
    data = fh.read(count)
    try:
        return json.loads(stripnull(data).decode())
    except ValueError as exc:
        logger().warning(f'<tifffile.read_json> raised {exc!r:.128}')
    return None


cdef dict read_mm_header(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
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


cdef object read_mm_stamp(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read FluoView mm_stamp tag value from file."""
    cdef str byteorder_str = '>' if byteorder == ByteOrder.MM else '<'
    return fh.read_array(byteorder_str + 'f8', 8)


cpdef dict read_uic1tag(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
    int64_t planecount = 0,
):
    """Read MetaMorph STK UIC1Tag value from file.

    Return empty dictionary if planecount is unknown.

    """
    if datatype not in {4, 5} or byteorder != ByteOrder.II:
        raise ValueError(f'invalid UIC1Tag {byteorder}{datatype}')
    result = {}
    if datatype == 5:
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


cdef dict read_uic2tag(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read MetaMorph STK UIC2Tag value from file."""
    if datatype != 5 or byteorder != ByteOrder.II:
        raise ValueError('invalid UIC2Tag')
    values = fh.read_array('<u4', 6 * count).reshape(count, 6)
    return {
        'ZDistance': values[:, 0] / values[:, 1],
        'DateCreated': values[:, 2],  # julian days
        'TimeCreated': values[:, 3],  # milliseconds
        'DateModified': values[:, 4],  # julian days
        'TimeModified': values[:, 5],  # milliseconds
    }


cdef dict read_uic3tag(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize
):
    """Read MetaMorph STK UIC3Tag value from file."""
    if datatype != 5 or byteorder != ByteOrder.II:
        raise ValueError('invalid UIC3Tag')
    values = fh.read_array('<u4', 2 * count).reshape(count, 2)
    return {'Wavelengths': values[:, 0] / values[:, 1]}


cdef dict read_uic4tag(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize
):
    """Read MetaMorph STK UIC4Tag value from file."""
    if datatype != 4 or byteorder != ByteOrder.II:
        raise ValueError('invalid UIC4Tag')
    result = {}
    while True:
        tagid: int = struct.unpack('<H', fh.read(2))[0]
        if tagid == 0:
            break
        name, value = read_uic_tag(fh, tagid, count, False)
        result[name] = value
    return result


cdef tuple read_uic_tag(
    FileHandle fh,
    int64_t tagid,
    int64_t planecount,
    bint offset
):
    """Read single UIC tag value from file and return tag name and value.

    UIC1Tags use an offset.

    """

    def read_int() -> int:
        return int(struct.unpack('<I', fh.read(4))[0])

    def read_int2() -> tuple[int, int]:
        value = struct.unpack('<2I', fh.read(8))
        return int(value[0]), (value[1])

    try:
        name, datatype = TIFF.UIC_TAGS[tagid]
    except IndexError:
        # unknown tag
        return f'_TagId{tagid}', read_int()

    Fraction = TIFF.UIC_TAGS[4][1]

    if offset:
        pos = fh.tell()
        if datatype not in {int, None}:
            off = read_int()
            if off < 8:
                # undocumented cases, or invalid offset
                if datatype is str:
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

    if datatype is None:
        # skip
        name = '_' + name
        value = read_int()
    elif datatype is int:
        # int
        value = read_int()
    elif datatype is Fraction:
        # fraction
        value = read_int2()
        value = value[0] / value[1]
    elif datatype is julian_datetime:
        # datetime
        value = read_int2()
        try:
            value = julian_datetime(*value)
        except Exception as exc:
            value = None
            logger().warning(
                f'<tifffile.read_uic_tag> reading {name} raised {exc!r:.128}'
            )
    elif datatype is read_uic_property:
        # ImagePropertyEx
        value = read_uic_property(fh)
    elif datatype is str:
        # pascal string
        size = read_int()
        if 0 <= size < 2**10:
            value = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
            value = bytes2str_stripnull(value)
        elif offset:
            value = ''
            logger().warning(
                f'<tifffile.read_uic_tag> invalid string in tag {name!r}'
            )
        else:
            raise ValueError(f'invalid string size {size}')
    elif planecount == 0:
        value = None
    elif datatype == '%ip':
        # sequence of pascal strings
        value = []
        for _ in range(planecount):
            size = read_int()
            if 0 <= size < 2**10:
                string = struct.unpack(f'{size}s', fh.read(size))[0][:-1]
                string = bytes2str_stripnull(string)
                value.append(string)
            elif offset:
                logger().warning(
                    f'<tifffile.read_uic_tag> invalid string in tag {name!r}'
                )
            else:
                raise ValueError(f'invalid string size: {size}')
    else:
        # struct or numpy type
        datatype = '<' + datatype
        if '%i' in datatype:
            datatype = datatype % planecount
        if '(' in datatype:
            # numpy type
            value = fh.read_array(datatype, 1)[0]
            if value.shape[-1] == 2:
                # assume fractions
                value = value[..., 0] / value[..., 1]
        else:
            # struct format
            value = struct.unpack(datatype, fh.read(struct.calcsize(datatype)))
            if len(value) == 1:
                value = value[0]

    if offset:
        fh.seek(pos + 4)

    return name, value


cdef dict read_uic_property(FileHandle fh):
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


cdef dict read_cz_lsminfo(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read CZ_LSMINFO tag value from file."""
    if byteorder != ByteOrder.II:
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

cdef dict read_sis(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read OlympusSIS structure from file.

    No specification is available. Only few fields are known.

    """
    result: dict[str, Any] = {}

    (magic, minute, hour, day, month, year, name, tagcount) = struct.unpack(
        '<4s6xhhhhh6x32sh', fh.read(60)
    )

    if magic != b'SIS0':
        raise ValueError('invalid OlympusSIS structure')

    result['name'] = bytes2str_stripnull(name)
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
            result['cameraname'] = bytes2str_stripnull(camname)
            result['picturetype'] = bytes2str_stripnull(pictype)
        elif tagtype == 10:
            # channel data
            continue
            # TODO: does not seem to work?
            # (length, _, exptime, emv, _, camname, _, mictype,
            #  ) = struct.unpack('<h22sId4s32s48s32s', fh.read(152))  # 720
            # result['exposuretime'] = exptime
            # result['emvoltage'] = emv
            # result['cameraname2'] = bytes2str_stripnull(camname)
            # result['microscopename'] = bytes2str_stripnull(mictype)

    return result

cdef tuple _keyindex(str key):
        # split key into name and index
        cdef int64_t index = 0
        cdef int64_t i = len(key.rstrip('0123456789'))
        if i < len(key):
            index = int(key[i:]) - 1
            key = key[:i]
        return key, index

cdef dict olympusini_metadata(str inistr):
    """Return OlympusSIS metadata from INI string.

    No specification is available.

    """

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
                    key, index = _keyindex(key)
                    result['ASD'][index][key] = value
            elif section_name == 'Band':
                if key[:3] == 'LUT':
                    lut = bands[iband]['LUT']
                    value = struct.pack('<I', value)
                    lut.append(
                        [ord(value[0:1]), ord(value[1:2]), ord(value[2:3])]
                    )
                else:
                    key, iband = _keyindex(key)
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

cdef dict read_sis_ini(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read OlympusSIS INI string from file."""
    inistr = bytes2str_stripnull(fh.read(count))
    try:
        return olympusini_metadata(inistr)
    except Exception as exc:
        logger().warning(f'<tifffile.olympusini_metadata> raised {exc!r:.128}')
        return {}


cdef dict read_tvips_header(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
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


cdef dict read_fei_metadata(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read FEI SFEG/HELIOS headers from file."""
    result: dict[str, Any] = {}
    section: dict[str, Any] = {}
    data = bytes2str_stripnull(fh.read(count))
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


cdef dict read_cz_sem(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
    """Read Zeiss SEM tag from file.

    See https://sourceforge.net/p/gwyddion/mailman/message/29275000/ for
    unnamed values.

    """
    result: dict[str, Any] = {'': ()}
    value: Any
    key = None
    data = bytes2str_stripnull(fh.read(count))
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


cdef dict read_nih_image_header(
    FileHandle fh,
    ByteOrder byteorder,
    int64_t datatype,
    int64_t count,
    int64_t offsetsize,
):
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
    if tag in {301, 320}:
        return read_colormap(fh, byteorder, dtype, count, offsetsize)
    elif tag in {33723, 37724, 33923, 40100, 50288, 50296, 50839, 65459}:
        return read_bytes(fh, byteorder, dtype, count, offsetsize)
    elif tag in {34363, 34368, 65426, 65432, 65439}: # NDPI unknown
        return read_numpy(fh, byteorder, dtype, count, offsetsize)
    elif tag in {34680, 34682}: # Helios NanoLab
        return read_fei_metadata(fh, byteorder, dtype, count, offsetsize)
    elif tag == 33628:
        return read_uic1tag(fh, byteorder, dtype, count, offsetsize, 0)  # Universal Imaging Corp STK
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
    elif tag == 51123:
        return read_json(fh, byteorder, dtype, count, offsetsize)
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

cdef inline bint readable_tag(int32_t tag) noexcept nogil:
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

cdef inline bint no_delay_load(int32_t tag) noexcept nogil:
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

cdef inline bint tag_is_tuple(int32_t tag) noexcept nogil:
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

cdef inline int64_t get_data_format_size(int64_t datatype) nogil:
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

cdef object interprete_data_format(int64_t datatype, const void* data, ByteOrder byteorder):
    """Interpret data according to TIFF DATATYPE and return as appropriate Python type.
    
    Parameters:
        datatype: TIFF DATATYPE value
        data: Pointer to binary data
        byteorder: Byte order (default is little-endian)
        
    Returns:
        Interpreted data value
    """
    cdef unsigned char* bytes_data = <unsigned char*>data
    cdef uint16_t* short_data = <uint16_t*>data
    cdef uint32_t* int_data = <uint32_t*>data
    cdef uint64_t* long_data = <uint64_t*>data
    cdef char* char_data = <char*>data
    cdef short* sshort_data = <short*>data
    cdef int* sint_data = <int*>data
    cdef long long* slong_data = <long long*>data
    cdef float* float_data = <float*>data
    cdef double* double_data = <double*>data
    cdef bint is_big_endian = byteorder == ByteOrder.MM
    
    if datatype == <int64_t>DATATYPE.BYTE:
        return bytes_data[0]
    elif datatype == <int64_t>DATATYPE.ASCII:
        return bytes_data[0:1]  # Return as bytes
    elif datatype == <int64_t>DATATYPE.SHORT:
        return _bswap16(short_data[0]) if is_big_endian else short_data[0]
    elif datatype == <int64_t>DATATYPE.LONG:
        return _bswap32(int_data[0]) if is_big_endian else int_data[0]
    elif datatype == <int64_t>DATATYPE.RATIONAL:
        if is_big_endian:
            return (_bswap32(int_data[0]), _bswap32(int_data[1]))
        else:
            return (int_data[0], int_data[1])
    elif datatype == <int64_t>DATATYPE.SBYTE:
        return char_data[0]
    elif datatype == <int64_t>DATATYPE.UNDEFINED:
        return bytes_data[0]
    elif datatype == <int64_t>DATATYPE.SSHORT:
        return <short>_bswap16(<uint16_t>sshort_data[0]) if is_big_endian else sshort_data[0]
    elif datatype == <int64_t>DATATYPE.SLONG:
        return <int>_bswap32(<uint32_t>sint_data[0]) if is_big_endian else sint_data[0]
    elif datatype == <int64_t>DATATYPE.SRATIONAL:
        if is_big_endian:
            return (<int>_bswap32(<uint32_t>sint_data[0]), <int>_bswap32(<uint32_t>sint_data[1]))
        else:
            return (sint_data[0], sint_data[1])
    elif datatype == <int64_t>DATATYPE.FLOAT:
        # Float handling requires special care for byte swapping
        if is_big_endian:
            # Would need an implementation of float byte swapping
            pass
        return float_data[0]
    elif datatype == <int64_t>DATATYPE.DOUBLE:
        if is_big_endian:
            # Would need an implementation of double byte swapping
            pass
        return double_data[0]
    elif datatype == <int64_t>DATATYPE.IFD:
        return _bswap32(int_data[0]) if is_big_endian else int_data[0]
    elif datatype == <int64_t>DATATYPE.LONG8:
        return _bswap64(long_data[0]) if is_big_endian else long_data[0]
    elif datatype == <int64_t>DATATYPE.SLONG8:
        return <long long>_bswap64(<uint64_t>slong_data[0]) if is_big_endian else slong_data[0]
    elif datatype == <int64_t>DATATYPE.IFD8:
        return _bswap64(long_data[0]) if is_big_endian else long_data[0]
    return ()  # Return empty tuple for unknown datatypes

@cython.final
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
    cdef TiffTag fromfile(
        FileHandle fh,
        TiffFormat tiff_format,
        int64_t offset,
        bytes header,
        bint validate
    ):
        """Return TiffTag instance from file.

        Parameters:
            fh:
                FileHandler instance tag is read from.
            tiff_format:
                TiffFormat instance that describes tag encoding
            offset:
                Position of tag structure in file.
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
            fh.read_at(offset, tiff_format.tagsize)

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
        elif (tiff_format.is_ndpi()
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
        tag._value = value
        tag.valueoffset = valueoffset
        return tag

    @staticmethod
    cdef TiffTag fromheader(FileHandle fh, TiffFormat tiff_format, TagHeader header):
        """Return TiffTag instance from parsed tag header.

        Parameters:
            fh:
                FileHandler instance tag is associated with.
            tiff_format:
                TiffFormat instance describing the file format.
            header:
                Parsed TagHeader structure containing tag information.

        Returns:
            TiffTag instance with properties set from header.
        """
        cdef int64_t i, valuesize
        cdef TiffTag tag = TiffTag.__new__(TiffTag)
        cdef object value = None
        tag.fh = fh
        tag.tiff_format = tiff_format
        tag.offset = 0  # Original file offset not preserved in header
        tag.code = header.code
        tag.datatype = header.datatype
        tag.count = header.count
        tag.valueoffset = header.as_offset

        # Check if this is an inlined value (fits within the tag's value field)
        try:
            valuesize = tag.count * get_data_format_size(tag.datatype)
            if valuesize <= tiff_format.tagoffsetthreshold and not readable_tag(tag.code):
                # For inlined values, we can use the pre-parsed values directly
                values = []
                for i in range(tag.count):
                    # Use the pre-parsed value
                    if tag.datatype in (1, 2, 3, 4, 6, 7, 8, 9, 13, 16, 17, 18): #integer formats
                        value = header.as_values[i].i
                    elif tag.datatype in (11, 12):  # FLOAT, DOUBLE
                        value = header.as_values[i].d
                    elif tag.datatype in (5, 10):  # RATIONAL, SRATIONAL
                        value = (header.as_values[2*i].i, header.as_values[2*i+1].i)
                    else:
                        raise ValueError(f"Unknown datatype {tag.datatype}")
                    values.append(value)

                # Process the value
                if tag.count > 1 or tag_is_tuple(header.code):
                    if tag.count > 1:
                        value = tuple(values)
                    else:
                        value = (values[0],)
                else:
                    value = values[0]
                if tag.datatype == 2:
                    # convert to bytes
                    value = bytes(values) # unsure if this is correct
                value = TiffTag._process_value(value, tag.code, tag.datatype, 0)
        except Exception as e:
            import traceback
            print("Error: ", e)
            print(traceback.format_exc())
            print("Code: ", tag.code)
            print("Datatype: ", tag.datatype)
            print("Count: ", tag.count)
            print("Valueoffset: ", tag.valueoffset)
            print(value)
            return None
        
        tag._value = value  # Value will be lazily loaded when requested for non inlined values
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
        cdef bytes data
        cdef const unsigned char* data_ptr
        cdef list result = []
        cdef int i

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
                #print("read7: ", value)
            except Exception as exc:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> raised {exc!r:.128}'
                )
            else:
                return value

        if datatype in [1, 2, 7]:
            # BYTES, ASCII, UNDEFINED
            value = fh.read_at(valueoffset, valuesize)
            #print("read8: ", value)
            if len(value) != valuesize:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> '
                    'could not read all values'
                )
        elif not(tag_is_tuple(code)) and count > 1024:
            fh.seek(valueoffset)
            value = read_numpy(fh, byteorder, datatype, count, offsetsize)
            #print("read9: ", value)
        else:
            data = fh.read_at(valueoffset, valuesize)
            data_ptr = data
        
            # Handle endianness appropriately
            if count == 1:
                # For single value, use interprete_data_format directly
                value = interprete_data_format(datatype, data_ptr, byteorder)
            else:
                # For multiple values, process each value separately
                result = []
                for i in range(count):
                    result.append(interprete_data_format(datatype, data_ptr + i * structsize, byteorder))
                value = tuple(result)
            #print("read10: ", value)
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
                value = bytes2str_stripnull_last(value)
            except UnicodeDecodeError as exc:
                logger().warning(
                    f'<tifffile.TiffTag {code} @{offset}> '
                    f'coercing invalid ASCII to bytes, due to {exc!r:.128}'
                )
            return value

        if code in {254, 255, 259, 262, 266, 274, 284, 296, 317, 338, 339}:
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

    cdef object value_get(self):
        """Value of tag, delay-loaded from file if necessary."""
        if self._value is None:
            # print(
            #     f'_read_value {self.code} {TIFF.TAGS.get(self.code)} '
            #     f'{self.datatype}[{self.count}] @{self.valueoffset} '
            # )
            #print("resolve: ", self.code)
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

    cdef void value_set(self, object value):
        self._value = value

    @property
    def value(self):
        return self.value_get()

    @value.setter
    def value(self, value):
        self.value_set(value)

    @property
    def name(self) -> str:
        """Name of tag from :py:attr:`_TIFF.TAGS` registry."""
        return TIFF.TAGS.get(self.code, str(self.code))

@cython.final
cdef class TiffTags:
    """Multidict-like interface to TiffTag instances in TiffPage.

    Differences to a regular dict:

    - values are instances of :py:class:`TiffTag`.
    - keys are :py:attr:`TiffTag.code` (int).
    - multiple values can be stored per key.
    - `iter()` returns values instead of keys.
    - `values()` and `items()` contain all values sorted by offset.
    - `len()` returns number of all values.
    - `get()` takes optional index argument.
    - some functions are not implemented, such as, `update` and `pop`.

    """

    def __cinit__(self, FileHandle fh, TiffFormat tiff):
        self._tags = []  # List to store all tags
        self._code_indices = cpp_multimap[int64_t, int]()  # Code -> indices multimap
        self.fh = fh
        self.tiff = tiff

    cdef bint contains_code(self, int64_t code) noexcept nogil:
        """Check if code is in map."""
        return self._code_indices.count(code) > 0

    def contains(self, object code):
        return self.contains_code(code)

    cdef int64_t _get_tag_value_as_int64(self, int index):
        """Retrieve a tag value as an int64"""
        # Check if header can directly be read as int64
        cdef int64_t datatype = self._headers[index].datatype
        cdef int64_t count = self._headers[index].count
        if count == 1:
            return self._headers[index].as_values[0].i
        return self._get_tag_value_at(index)

    cdef double _get_tag_value_as_double(self, int index):
        """Retrieve a tag value as a double"""
        # Check if header can directly be read as double
        cdef int64_t datatype = self._headers[index].datatype
        cdef int64_t count = self._headers[index].count
        if count == 1:
            return self._headers[index].as_values[0].d
        return self._get_tag_value_at(index)

    cdef object _get_tag_value_at(self, int index):
        """Retrieve the tag value at the target index"""
        if self._tags[index] is not None:
            return self._tags[index].value_get()
        # Create TiffTag instance
        cdef TiffTag tag = TiffTag.fromheader(self.fh, self.tiff, self._headers[index])
        self._tags[index] = tag
        if tag is None:
            return None
        return tag.value_get()

    cdef TiffTag _get_tag_at(self, int index):
        """Retrieve the TiffTag at the target index"""
        if self._tags[index] is not None:
            return self._tags[index]
        # Create TiffTag instance
        cdef TiffTag tag = TiffTag.fromheader(self.fh, self.tiff, self._headers[index])
        self._tags[index] = tag
        return tag

    cdef void load_tags(self, bytes data):
        """
        load the tags contained in data.
        """

        cdef vector[TagHeader] headers = vector[TagHeader]()
        cdef TagHeader header
        cdef unsigned char *data_ptr = <unsigned char *>data
        cdef int64_t data_len = len(data)
        cdef pair[int64_t, int] element
        with nogil:
            self.tiff.parse_tag_headers(headers, data_ptr, data_len)

        assert not self.tiff.is_ndpi() # NDPI format is not supported yet

        # process the headers
        for header in headers:
            element.first = header.code
            element.second = len(self._tags)
            self._code_indices.insert(element)
            self._tags.append(None) # Lazy loading
            self._headers.push_back(header)


    cpdef list keys(self):
        """Return codes of all tags."""
        cdef list result = []
        cdef TagHeader header

        for header in self._headers:
            result.append(header.code)
                
        return result

    cpdef list values(self):
        """Return all tags in order they are stored in file."""
        cdef list result = []
        cdef int index
        
        for index in range(<int>self._headers.size()):
            result.append(self._get_tag_value_at(index))
                
        return result

    cpdef list items(self):
        """Return all (code, tag) pairs in order tags as stored in file."""
        cdef list result = []
        cdef TiffTag tag
        cdef int index

        for index in range(<int>self._headers.size()):
            tag = self._get_tag_at(index)
            result.append((tag.code, tag))
                
        return result

    cpdef object valueof(
        self,
        int64_t code,
        object default=None,
        int64_t index=0,
    ):
        """Return value of tag by code if exists, else default.

        Parameters:
            key:
                Code of tag to return.
            default:
                Another value to return if specified tag is corrupted or
                not found.
            index:
                Specifies tag in case of multiple tags with identical code.
                The default is the first tag.

        """
        cdef pair[cpp_multimap[int64_t, int].iterator, cpp_multimap[int64_t, int].iterator] m_range
        cdef cpp_multimap[int64_t, int].iterator it
        cdef TiffTag tag
        cdef int idx
        cdef int i

        if index == 0:  # Most common case: get first tag
            # Find first non-None tag
            it = self._code_indices.find(code)
            if it == self._code_indices.end():
                return default
            idx = dereference(it).second
            return self._get_tag_value_at(idx)
        else:
            if self._code_indices.count(code) == 0:
                return default
            m_range = self._code_indices.equal_range(code)
            it = m_range.first
            # Advance iterator to the requested index
            i = 0
            while i < index and it != m_range.second:
                postincrement(it)
                i += 1
                
            # If we reached the end before the requested index or no tag at index
            if it == m_range.second:
                return default
                
            # Get the tag at the requested index
            idx = dereference(it).second
            return self._get_tag_value_at(idx)

    cdef int64_t valueof_int(
        self,
        int64_t code,
        int64_t default,
        int64_t index):
        """Return int value of tag by code if exists, else default.

        Parameters:
            code:
                Code of tag to return.
            default:
                Another value to return if specified tag is corrupted or
                not found.
            index:
                Specifies tag in case of multiple tags with identical code.
                The default is the first tag.

        """
        cdef pair[cpp_multimap[int64_t, int].iterator, cpp_multimap[int64_t, int].iterator] m_range
        cdef cpp_multimap[int64_t, int].iterator it
        cdef TiffTag tag
        cdef int idx
        cdef int i

        if index == 0:  # Most common case: get first tag
            # Find first non-None tag
            it = self._code_indices.find(code)
            if it == self._code_indices.end():
                return default
            idx = dereference(it).second
            return self._get_tag_value_as_int64(idx)
        else:
            if self._code_indices.count(code) == 0:
                return default
            m_range = self._code_indices.equal_range(code)
            it = m_range.first
            # Advance iterator to the requested index
            i = 0
            while i < index and it != m_range.second:
                postincrement(it)
                i += 1
                
            # If we reached the end before the requested index or no tag at index
            if it == m_range.second:
                return default
                
            # Get the tag at the requested index
            idx = dereference(it).second
            return self._get_tag_value_as_int64(idx)

    cdef double valueof_double(
        self,
        int64_t code,
        double default,
        int64_t index):
        """Return double value of tag by code if exists, else default.

        Parameters:
            code:
                Code of tag to return.
            default:
                Another value to return if specified tag is corrupted or
                not found.
            index:
                Specifies tag in case of multiple tags with identical code.
                The default is the first tag.

        """
        cdef pair[cpp_multimap[int64_t, int].iterator, cpp_multimap[int64_t, int].iterator] m_range
        cdef cpp_multimap[int64_t, int].iterator it
        cdef TiffTag tag
        cdef int idx
        cdef int i

        if index == 0:  # Most common case: get first tag
            # Find first non-None tag
            it = self._code_indices.find(code)
            if it == self._code_indices.end():
                return default
            idx = dereference(it).second
            return self._get_tag_value_as_double(idx)
        else:
            if self._code_indices.count(code) == 0:
                return default
            m_range = self._code_indices.equal_range(code)
            it = m_range.first
            # Advance iterator to the requested index
            i = 0
            while i < index and it != m_range.second:
                postincrement(it)
                i += 1
                
            # If we reached the end before the requested index or no tag at index
            if it == m_range.second:
                return default
                
            # Get the tag at the requested index
            idx = dereference(it).second
            return self._get_tag_value_as_double(idx)

    cpdef TiffTag get(
        self,
        int64_t code,
        TiffTag default=None,
        int64_t index=0):
        """Return tag by code if exists, else default.

        Parameters:
            code:
                Code of tag to return.
            default:
                Another tag to return if specified tag is corrupted or
                not found.
            index:
                Specifies tag in case of multiple tags with identical code.
                The default is the first tag.

        """
        cdef pair[cpp_multimap[int64_t, int].iterator, cpp_multimap[int64_t, int].iterator] m_range
        cdef cpp_multimap[int64_t, int].iterator it
        cdef TiffTag tag
        cdef int idx
        cdef int i

        if index == 0:  # Most common case: get first tag
            # Find first non-None tag
            it = self._code_indices.find(code)
            if it == self._code_indices.end():
                return default
            idx = dereference(it).second
            return self._get_tag_at(idx)
        else:
            if self._code_indices.count(code) == 0:
                return default
            m_range = self._code_indices.equal_range(code)
            it = m_range.first
            # Advance iterator to the requested index
            i = 0
            while i < index and it != m_range.second:
                postincrement(it)
                i += 1
                
            # If we reached the end before the requested index or no tag at index
            if it == m_range.second:
                return default
                
            # Get the tag at the requested index
            idx = dereference(it).second
            return self._get_tag_at(idx)

    cdef int64_t get_count(self, int64_t code, int64_t index) noexcept nogil:
        """Retrieve the header count for target code. 0 if not present"""
        cdef pair[cpp_multimap[int64_t, int].iterator, cpp_multimap[int64_t, int].iterator] m_range
        cdef cpp_multimap[int64_t, int].iterator it
        cdef int idx
        cdef int i

        if index == 0:  # Most common case: get first tag
            # Find first non-None tag
            it = self._code_indices.find(code)
            if it == self._code_indices.end():
                return 0
            idx = dereference(it).second
            return self._headers[idx].count
        else:
            if self._code_indices.count(code) == 0:
                return 0
            m_range = self._code_indices.equal_range(code)
            it = m_range.first
            # Advance iterator to the requested index
            i = 0
            while i < index and it != m_range.second:
                postincrement(it)
                i += 1
                
            # If we reached the end before the requested index or no tag at index
            if it == m_range.second:
                return 0
                
            # Get the tag at the requested index
            idx = dereference(it).second
            return self._headers[idx].count



@cython.final
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

    def __init__(self, object arg):
        self._entries = []  # Store (code, name) tuples
        self._code_indices = cpp_multimap[int64_t, int]()
        self._name_indices = cpp_multimap[cpp_string, int]()
        self._entry_count = 0
        self.update(arg)

    cdef bint _contains_code(self, int64_t code) nogil:
        """Check if code is in registry."""
        return self._code_indices.count(code) > 0

    cdef bint _contains_name(self, cpp_string name) nogil:
        """Check if name is in registry."""
        return self._name_indices.count(name) > 0
        
    cdef vector[int] _get_code_indices(self, int64_t code) nogil:
        """Get indices for a code."""
        cdef vector[int] result = vector[int]()
        cdef pair[cpp_multimap[int64_t, int].iterator, cpp_multimap[int64_t, int].iterator] range
        cdef cpp_multimap[int64_t, int].iterator it
        
        if self._code_indices.count(code) > 0:
            range = self._code_indices.equal_range(code)
            it = range.first
            while it != range.second:
                result.push_back(dereference(it).second)
                postincrement(it)
        
        return result
        
    cdef vector[int] _get_name_indices(self, cpp_string name) nogil:
        """Get indices for a name."""
        cdef vector[int] result = vector[int]()
        cdef pair[cpp_multimap[cpp_string, int].iterator, cpp_multimap[cpp_string, int].iterator] range
        cdef cpp_multimap[cpp_string, int].iterator it
        
        if self._name_indices.count(name) > 0:
            range = self._name_indices.equal_range(name)
            it = range.first
            while it != range.second:
                result.push_back(dereference(it).second)
                postincrement(it)
        
        return result
        
    cdef void _add_index(self, int64_t code, str name, int index) nogil:
        """Add an index to code and name maps."""
        cdef cpp_string name_bytes
        
        with gil:
            name_bytes = name.encode('utf8')
        
        self._code_indices.insert(pair[int64_t, int](code, index))
        self._name_indices.insert(pair[cpp_string, int](name_bytes, index))

    cpdef bint contains(self, object item):
        """Check if item is in registry."""
        cdef int64_t code
        cdef cpp_string name_bytes
        
        if isinstance(item, int):
            code = item
            return self._contains_code(code)
        elif isinstance(item, str):
            name_bytes = item.encode('utf8')
            return self._contains_name(name_bytes)
        else:
            return False

    cpdef void update(self, object arg):
        """Add mapping of codes to names to registry.

        Parameters:
            arg: Mapping of codes to names.
        """
        cdef int64_t code
        cdef str name
        
        if isinstance(arg, TiffTagRegistry):
            # Copy entries from other registry
            for code, name in arg.items():
                self.add(code, name)
            return
            
        if isinstance(arg, dict):
            arg = list(arg.items())
            
        for code, name in arg:
            self.add(code, name)

    cpdef void add(self, int64_t code, str name):
        """Add code and name to registry."""
        cdef cpp_string name_bytes = name.encode('utf8')
        cdef int64_t i, idx
        cdef vector[int] code_indices, name_indices
        cdef tuple entry
        
        # Check if already exists
        if self._contains_code(code) and self._contains_name(name_bytes):
            code_indices = self._get_code_indices(code)
            name_indices = self._get_name_indices(name_bytes)
            
            # Check if this exact pair already exists
            for i in range(<int64_t>code_indices.size()):
                idx = code_indices[i]
                entry = self._entries[idx]
                if entry is not None and entry[0] == code and entry[1] == name:
                    return  # Entry already exists
        
        # Add new entry
        idx = len(self._entries)
        self._entries.append((code, name))
        self._add_index(code, name, idx)
        self._entry_count += 1

    @staticmethod
    cdef int64_t _code_key(self, tuple element):
        return element[0]

    cpdef list items(self):
        """Return all registry items as (code, name)."""
        cdef list result = []
        cdef tuple entry
        
        for entry in self._entries:
            if entry is not None:
                result.append(entry)
                
        return sorted(result, key=TiffTagRegistry._code_key)

    cpdef object get(self, object key, object default=None):
        """Return first code or name if exists, else default.

        Parameters:
            key: tag code or name to lookup.
            default: value to return if key is not found.
        """
        cdef int64_t code
        cdef cpp_string name_bytes
        cdef vector[int] indices
        cdef int64_t i, idx
        cdef tuple entry
        
        if isinstance(key, int):
            code = key
            if not self._contains_code(code):
                return default
                
            indices = self._get_code_indices(code)
            for i in range(<int64_t>indices.size()):
                idx = indices[i]
                entry = self._entries[idx]
                if entry is not None:
                    return entry[1]  # Return name
                    
        elif isinstance(key, str):
            name_bytes = key.encode('utf8')
            if not self._contains_name(name_bytes):
                return default
                
            indices = self._get_name_indices(name_bytes)
            for i in range(<int64_t>indices.size()):
                idx = indices[i]
                entry = self._entries[idx]
                if entry is not None:
                    return entry[0]  # Return code
                    
        return default

    cpdef object getall(self, object key, object default=None):
        """Return list of all codes or names if exists, else default.

        Parameters:
            key: tag code or name to lookup.
            default: value to return if key is not found.
        """
        cdef list result = []
        cdef int64_t code
        cdef cpp_string name_bytes
        cdef vector[int] indices
        cdef int64_t i, idx
        cdef tuple entry
        
        if isinstance(key, int):
            code = key
            if not self._contains_code(code):
                return default
                
            indices = self._get_code_indices(code)
            for i in range(<int64_t>indices.size()):
                idx = indices[i]
                entry = self._entries[idx]
                if entry is not None:
                    result.append(entry[1])  # Collect names
                    
        elif isinstance(key, str):
            name_bytes = key.encode('utf8')
            if not self._contains_name(name_bytes):
                return default
                
            indices = self._get_name_indices(name_bytes)
            for i in range(<int64_t>indices.size()):
                idx = indices[i]
                entry = self._entries[idx]
                if entry is not None:
                    result.append(entry[0])  # Collect codes
                    
        return result if result else default

    def __getitem__(self, object key):
        """Return first code or name. Raise KeyError if not found."""
        cdef result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __delitem__(self, object key):
        """Delete all tags of code or name."""
        cdef bint found = False
        cdef int64_t code
        cdef cpp_string name_bytes
        cdef vector[int] indices
        cdef int64_t i, idx
        cdef tuple entry

        assert(False) # incorrect

        if isinstance(key, int):
            code = key
            if not self._contains_code(code):
                raise KeyError(key)

            indices = self._get_code_indices(code)
            for i in range(<int64_t>indices.size()):
                idx = indices[i]
                if self._entries[idx] is not None:
                    self._entries[idx] = None
                    found = True
                    self._entry_count -= 1
            
            # Remove code from the multimap
            self._code_indices.erase(code)
                
        elif isinstance(key, str):
            name_bytes = bytes(key, 'utf8')
            if not self._contains_name(name_bytes):
                raise KeyError(key)

            indices = self._get_name_indices(name_bytes)
            for i in range(<int64_t>indices.size()):
                idx = indices[i]
                if self._entries[idx] is not None:
                    self._entries[idx] = None
                    found = True
                    self._entry_count -= 1

            # Remove name from the multimap
            self._name_indices.erase(name_bytes)
                
        if not found:
            raise KeyError(key)

    def __contains__(self, object item):
        """Return if code or name is in registry."""
        return self.contains(item)

    def __iter__(self):
        """Return iterator over all items in registry."""
        return iter(self.items())

    def __len__(self):
        """Return number of registered tags."""
        return self._entry_count
