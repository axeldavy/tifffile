#cython: language_level=3
#cython: boundscheck=True
#cython: wraparound=True
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

import binascii
from datetime import datetime as DateTime
import io
import logging
import math
import numpy
import re

from .types import PHOTOMETRIC

cdef void lock_gil_friendly_block(unique_lock[recursive_mutex] &m) noexcept:
    """
    Same as lock_gil_friendly, but blocks until the job is done.
    We inline the fast path, but not this one as it generates
    more code.
    """
    # Release the gil to enable python processes eventually
    # holding the lock to run and release it.
    # Block until we get the lock
    cdef bint locked = False
    while not(locked):
        with nogil:
            # Block until the mutex is released
            m.lock()
            # Unlock to prevent deadlock if another
            # thread holding the gil requires m
            # somehow
            m.unlock()
        locked = m.try_lock()

def apply_colormap(
    image: NDArray[Any], colormap: NDArray[Any], contig: bool = True
) -> NDArray[Any]:
    """Return palette-colored image.

    The image array values are used to index the colormap on axis 1.
    The returned image array is of shape `image.shape+colormap.shape[0]`
    and dtype `colormap.dtype`.

    Parameters:
        image:
            Array of indices into colormap.
        colormap:
            RGB lookup table aka palette of shape `(3, 2**bitspersample)`.
        contig:
            Return contiguous array.

    Examples:
        >>> im = numpy.arange(256, dtype='uint8')
        >>> colormap = numpy.vstack([im, im, im]).astype('uint16') * 256
        >>> apply_colormap(im, colormap)[-1]
        array([65280, 65280, 65280], dtype=uint16)

    """
    image = numpy.take(colormap, image, axis=1)
    image = numpy.rollaxis(image, 0, image.ndim)
    if contig:
        image = numpy.ascontiguousarray(image)
    return image

def asbool(
    value: str | bytes,
    /,
    true: Sequence[str | bytes] | None = None,
    false: Sequence[str | bytes] | None = None,
) -> bool | bytes:
    """Return string as bool if possible, else raise TypeError.

    >>> asbool(b' False ')
    False
    >>> asbool('ON', ['on'], ['off'])
    True

    """
    value = value.strip().lower()
    isbytes = False
    if true is None:
        if isinstance(value, bytes):
            if value == b'true':
                return True
            isbytes = True
        elif value == 'true':
            return True
    elif value in true:
        return True
    if false is None:
        if isbytes or isinstance(value, bytes):
            if value == b'false':
                return False
        elif value == 'false':
            return False
    elif value in false:
        return False
    raise TypeError

def astype(value: Any, types: Sequence[Any] | None = None) -> Any:
    """Return argument as one of types if possible.

    >>> astype('42')
    42
    >>> astype('3.14')
    3.14
    >>> astype('True')
    True
    >>> astype(b'Neee-Wom')
    'Neee-Wom'

    """
    if types is None:
        types = int, float, asbool, bytes2str
    for typ in types:
        try:
            return typ(value)
        except (ValueError, AttributeError, TypeError, UnicodeEncodeError):
            pass
    return value

def bytes2str(
    b: bytes, encoding: str | None = None, errors: str = 'strict'
) -> str:
    """Return Unicode string from encoded bytes."""
    if encoding is not None:
        return b.decode(encoding, errors)
    try:
        return b.decode('utf-8', errors)
    except UnicodeDecodeError:
        return b.decode('cp1252', errors)


def bytestr(s: str | bytes, encoding: str = 'cp1252') -> bytes:
    """Return bytes from Unicode string, else pass through."""
    return s.encode(encoding) if isinstance(s, str) else s


def clean_whitespace(string: str, compact: bool = False) -> str:
    r"""Return string with compressed whitespace.

    >>> clean_whitespace('  a  \n\n  b ')
    'a\n b'

    """
    string = (
        string.replace('\r\n', '\n')
        .replace('\r', '\n')
        .replace('\n\n', '\n')
        .replace('\t', ' ')
        .replace('  ', ' ')
        .replace('  ', ' ')
        .replace(' \n', '\n')
    )
    if compact:
        string = (
            string.replace('\n', ' ')
            .replace('[ ', '[')
            .replace('  ', ' ')
            .replace('  ', ' ')
            .replace('  ', ' ')
        )
    return string.strip()


def create_output(
    out: OutputType,
    /,
    shape: Sequence[int],
    dtype: DTypeLike,
    *,
    mode: Literal['r+', 'w+', 'r', 'c'] = 'w+',
    suffix: str | None = None,
    fillvalue: int | float | None = 0,
) -> NDArray[Any] | numpy.memmap[Any, Any]:
    """Return NumPy array where images of shape and dtype can be copied.

    Parameters:
        out:
            Specifies kind of array to return:

                `None`:
                    A new array of shape and dtype is created and returned.
                `numpy.ndarray`:
                    An existing, writable array compatible with `dtype` and
                    `shape`. A view of the array is returned.
                `'memmap'` or `'memmap:tempdir'`:
                    A memory-map to an array stored in a temporary binary file
                    on disk is created and returned.
                `str` or open file:
                    File name or file object used to create a memory-map
                    to an array stored in a binary file on disk.
                    The memory-mapped array is returned.
        shape:
            Shape of NumPy array to return.
        dtype:
            Data type of NumPy array to return.
        suffix:
            Suffix of `NamedTemporaryFile` if `out` is 'memmap'.
            The default suffix is 'memmap'.
        fillvalue:
            Value to initialize newly created arrays.
            If *None*, return an uninitialized array.

    """
    shape = tuple(shape)
    if out is None:
        if fillvalue is None:
            return numpy.empty(shape, dtype)
        if fillvalue:
            out = numpy.empty(shape, dtype)
            out[:] = fillvalue
            return out
        return numpy.zeros(shape, dtype)
    if isinstance(out, numpy.ndarray):
        if product(shape) != product(out.shape):
            raise ValueError('incompatible output shape')
        if not numpy.can_cast(dtype, out.dtype):
            raise ValueError('incompatible output dtype')
        return out.reshape(shape)
    if isinstance(out, str) and out[:6] == 'memmap':
        import tempfile

        tempdir = out[7:] if len(out) > 7 else None
        if suffix is None:
            suffix = '.memmap'
        with tempfile.NamedTemporaryFile(dir=tempdir, suffix=suffix) as fh:
            out = numpy.memmap(fh, shape=shape, dtype=dtype, mode=mode)
            if fillvalue:
                out[:] = fillvalue
            return out
    out = numpy.memmap(out, shape=shape, dtype=dtype, mode=mode)
    if fillvalue:
        out[:] = fillvalue
    return out

def enumstr(enum: Any, /) -> str:
    """Return short string representation of Enum member.

    >>> enumstr(PHOTOMETRIC.RGB)
    'RGB'

    """
    name = enum.name
    if name is None:
        name = str(enum)
    return name


def enumarg(enum: type[enum.IntEnum], arg: Any, /) -> enum.IntEnum:
    """Return enum member from its name or value.

    Parameters:
        enum: Type of IntEnum.
        arg: Name or value of enum member.

    Returns:
        Enum member matching name or value.

    Raises:
        ValueError: No enum member matches name or value.

    Examples:
        >>> enumarg(PHOTOMETRIC, 2)
        <PHOTOMETRIC.RGB: 2>
        >>> enumarg(PHOTOMETRIC, 'RGB')
        <PHOTOMETRIC.RGB: 2>

    """
    try:
        return enum(arg)
    except Exception:
        try:
            return enum[arg.upper()]
        except Exception as exc:
            raise ValueError(f'invalid argument {arg!r}') from exc

def hexdump(
    data: bytes,
    /,
    *,
    width: int = 75,
    height: int = 24,
    snipat: int | float | None = 0.75,
    modulo: int = 2,
    ellipsis: str | None = None,
) -> str:
    """Return hexdump representation of bytes.

    Parameters:
        data:
            Bytes to represent as hexdump.
        width:
            Maximum width of hexdump.
        height:
            Maximum number of lines of hexdump.
        snipat:
            Approximate position at which to split long hexdump.
        modulo:
            Number of bytes represented in line of hexdump are modulus
            of this value.
        ellipsis:
            Characters to insert for snipped content of long hexdump.
            The default is '...'.

    Examples:
        >>> hexdump(binascii.unhexlify('49492a00080000000e00fe0004000100'))
        '49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 II*.............'

    """
    size = len(data)
    if size < 1 or width < 2 or height < 1:
        return ''
    if height == 1:
        addr = b''
        bytesperline = min(
            modulo * (((width - len(addr)) // 4) // modulo), size
        )
        if bytesperline < 1:
            return ''
        nlines = 1
    else:
        addr = <bytes>(b'%%0%ix: ' % len(b'%x' % size))
        bytesperline = min(
            modulo * (((width - len(addr % 1)) // 4) // modulo), size
        )
        if bytesperline < 1:
            return ''
        width = 3 * bytesperline + len(addr % 1)
        nlines = (size - 1) // bytesperline + 1

    if snipat is None or snipat == 1:
        snipat = height
    elif 0 < abs(snipat) < 1:
        snipat = int(math.floor(height * snipat))
    if snipat < 0:
        snipat += height
    assert isinstance(snipat, int)

    blocks: list[tuple[int, bytes | None]]

    if height == 1 or nlines == 1:
        blocks = [(0, data[:bytesperline])]
        addr = b''
        height = 1
        width = 3 * bytesperline
    elif not height or nlines <= height:
        blocks = [(0, data)]
    elif snipat <= 0:
        start = bytesperline * (nlines - height)
        blocks = [(start, data[start:])]  # (start, None)
    elif snipat >= height or height < 3:
        end = bytesperline * height
        blocks = [(0, data[:end])]  # (end, None)
    else:
        end1 = bytesperline * snipat
        end2 = bytesperline * (height - snipat - 2)
        if size % bytesperline:
            end2 += size % bytesperline
        else:
            end2 += bytesperline
        blocks = [
            (0, data[:end1]),
            (size - end1 - end2, None),
            (size - end2, data[size - end2 :]),
        ]

    if ellipsis is None:
        if addr and bytesperline > 3:
            elps = b' ' * (len(addr % 1) + bytesperline // 2 * 3 - 2)
            elps += b'...'
        else:
            elps = b'...'
    else:
        elps = ellipsis.encode('cp1252')

    result = []
    for start, bstr in blocks:
        if bstr is None:
            result.append(elps)  # 'skip %i bytes' % start)
            continue
        hexstr = binascii.hexlify(bstr)
        strstr = re.sub(br'[^\x20-\x7f]', b'.', bstr)
        for i in range(0, len(bstr), bytesperline):
            h = hexstr[2 * i : 2 * i + bytesperline * 2]
            r = (addr % (i + start)) if height > 1 else addr
            r += b' '.join(h[i : i + 2] for i in range(0, 2 * bytesperline, 2))
            r += b' ' * (width - len(r))
            r += strstr[i : i + bytesperline]
            result.append(r)
    return b'\n'.join(result).decode('ascii')

def identityfunc(arg: Any, *args: Any, **kwargs: Any) -> Any:
    """Single argument identity function.

    >>> identityfunc('arg')
    'arg'

    """
    return arg

def indent(*args: Any) -> str:
    """Return joined string representations of objects with indented lines.

    >>> print(indent('Title:', 'Text'))
    Title:
      Text

    """
    text = '\n'.join(str(arg) for arg in args)
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]

def isprintable(string: str | bytes, /) -> bool:
    r"""Return if all characters in string are printable.

    >>> isprintable('abc')
    True
    >>> isprintable(b'\01')
    False

    """
    string = string.strip()
    if not string:
        return True
    try:
        return string.isprintable()  # type: ignore[union-attr]
    except Exception:
        pass
    try:
        return string.decode().isprintable()  # type: ignore[union-attr]
    except Exception:
        pass
    return False

def julian_datetime(julianday: int, millisecond: int = 0, /) -> DateTime:
    """Return datetime from days since 1/1/4713 BC and ms since midnight.

    Convert Julian dates according to MetaMorph.

    >>> julian_datetime(2451576, 54362783)
    datetime.datetime(2000, 2, 2, 15, 6, 2, 783000)

    """
    if julianday <= 1721423:
        # return DateTime.min  # ?
        raise ValueError(f'no datetime before year 1 ({julianday=})')

    a = julianday + 1
    if a > 2299160:
        alpha = math.trunc((a - 1867216.25) / 36524.25)
        a += 1 + alpha - alpha // 4
    b = a + (1524 if a > 1721423 else 1158)
    c = math.trunc((b - 122.1) / 365.25)
    d = math.trunc(365.25 * c)
    e = math.trunc((b - d) / 30.6001)

    day = b - d - math.trunc(30.6001 * e)
    month = e - (1 if e < 13.5 else 13)
    year = c - (4716 if month > 2.5 else 4715)

    hour, millisecond = divmod(millisecond, 1000 * 60 * 60)
    minute, millisecond = divmod(millisecond, 1000 * 60)
    second, millisecond = divmod(millisecond, 1000)

    return DateTime(year, month, day, hour, minute, second, millisecond * 1000)

def jpeg_decode_colorspace(
    photometric: int,
    planarconfig: int,
    extrasamples: tuple[int, ...],
    jfif: bool,
    /,
) -> tuple[int | None, int | str | None]:
    """Return JPEG and output color space for `jpeg_decode` function."""
    colorspace: int | None = None
    outcolorspace: int | str | None = None
    if extrasamples:
        pass
    elif photometric == 6:
        # YCBCR -> RGB
        outcolorspace = 2  # RGB
    elif photometric == 2:
        # RGB -> RGB
        if not jfif:
            # found in Aperio SVS
            colorspace = 2
        outcolorspace = 2
    elif photometric == 5:
        # CMYK
        outcolorspace = 4
    elif photometric > 3:
        outcolorspace = PHOTOMETRIC(photometric).name
    if planarconfig != 1:
        outcolorspace = 1  # decode separate planes to grayscale
    return colorspace, outcolorspace

def logger() -> logging.Logger:
    """Return logging.getLogger('tifffile')."""
    return logging.getLogger(__name__.replace('tifffile.tifffile', 'tifffile'))

def natural_sorted(iterable: Iterable[str], /) -> list[str]:
    """Return human-sorted list of strings.

    Use to sort file names.

    >>> natural_sorted(['f1', 'f2', 'f10'])
    ['f1', 'f2', 'f10']

    """

    def sortkey(x: str, /) -> list[int | str]:
        return [(int(c) if c.isdigit() else c) for c in re.split(numbers, x)]

    numbers = re.compile(r'(\d+)')
    return sorted(iterable, key=sortkey)

def pformat(
    arg: Any,
    /,
    *,
    height: int | None = 24,
    width: int | None = 79,
    linewidth: int | None = 288,
    compact: bool = True,
) -> str:
    """Return pretty formatted representation of object as string.

    Whitespace might be altered. Long lines are cut off.

    """
    if height is None or height < 1:
        height = 1024
    if width is None or width < 1:
        width = 256
    if linewidth is None or linewidth < 1:
        linewidth = width

    npopt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=100, linewidth=width)

    if isinstance(arg, bytes):
        if arg[:5].lower() == b'<?xml' or arg[-4:] == b'OME>':
            arg = bytes2str(arg)

    if isinstance(arg, bytes):
        if isprintable(arg):
            arg = bytes2str(arg)
            arg = clean_whitespace(arg)
        else:
            numpy.set_printoptions(**npopt)
            return hexdump(arg, width=width, height=height, modulo=1)
        arg = arg.rstrip()
    elif isinstance(arg, str):
        if arg[:5].lower() == '<?xml' or arg[-4:] == 'OME>':
            arg = arg[: 4 * width] if height == 1 else pformat_xml(arg)
        # too slow
        # else:
        #    import textwrap
        #    return '\n'.join(
        #        textwrap.wrap(arg, width=width, max_lines=height, tabsize=2)
        #    )
        arg = arg.rstrip()
    elif isinstance(arg, numpy.record):
        arg = arg.pprint()
    else:
        import pprint

        arg = pprint.pformat(arg, width=width, compact=compact)

    numpy.set_printoptions(**npopt)

    if height == 1:
        arg = arg[: width * width]
        arg = clean_whitespace(arg, compact=True)
        return arg[:linewidth]

    argl = list(arg.splitlines())
    if len(argl) > height:
        arg = '\n'.join(
            line[:linewidth]
            for line in argl[: height // 2] + ['...'] + argl[-height // 2 :]
        )
    else:
        arg = '\n'.join(line[:linewidth] for line in argl[:height])
    return arg


def pformat_xml(xml: str | bytes, /) -> str:
    """Return pretty formatted XML."""
    try:
        from lxml import etree

        if not isinstance(xml, bytes):
            xml = xml.encode()
        tree = etree.parse(io.BytesIO(xml))
        xml = etree.tostring(
            tree,
            pretty_print=True,
            xml_declaration=True,
            encoding=tree.docinfo.encoding,
        )
        assert isinstance(xml, bytes)
        xml = bytes2str(xml)
    except Exception:
        if isinstance(xml, bytes):
            xml = bytes2str(xml)
        xml = xml.replace('><', '>\n<')
    return xml.replace('  ', ' ').replace('\t', ' ')

def recarray2dict(recarray: numpy.recarray[Any, Any], /) -> dict[str, Any]:
    """Return numpy.recarray as dictionary.

    >>> r = numpy.array(
    ...     [(1.0, 2, 'a'), (3.0, 4, 'bc')],
    ...     dtype=[('x', '<f4'), ('y', '<i4'), ('s', 'S2')],
    ... )
    >>> recarray2dict(r)
    {'x': [1.0, 3.0], 'y': [2, 4], 's': ['a', 'bc']}
    >>> recarray2dict(r[1])
    {'x': 3.0, 'y': 4, 's': 'bc'}

    """
    # TODO: subarrays
    value: Any
    result = {}
    for descr in recarray.dtype.descr:
        name, dtype = descr[:2]
        value = recarray[name]
        if value.ndim == 0:
            value = value.tolist()
            if dtype[1] == 'S':
                value = bytes2str(stripnull(value))
        elif value.ndim == 1:
            value = value.tolist()
            if dtype[1] == 'S':
                value = [bytes2str(stripnull(v)) for v in value]
        result[name] = value
    return result

def snipstr(
    string: str,
    /,
    width: int = 79,
    *,
    snipat: int | float | None = None,
    ellipsis: str | None = None,
) -> str:
    """Return string cut to specified length.

    Parameters:
        string:
            String to snip.
        width:
            Maximum length of returned string.
        snipat:
            Approximate position at which to split long strings.
            The default is 0.5.
        ellipsis:
            Characters to insert between splits of long strings.
            The default is '...'.

    Examples:
        >>> snipstr('abcdefghijklmnop', 8)
        'abc...op'

    """
    if snipat is None:
        snipat = 0.5
    if ellipsis is None:
        if isinstance(string, bytes):  # type: ignore[unreachable]
            ellipsis = b'...'
        else:
            ellipsis = '\u2026'
    esize = len(ellipsis)

    splitlines = string.splitlines()
    # TODO: finish and test multiline snip

    result = []
    for line in splitlines:
        if line is None:
            result.append(ellipsis)
            continue
        linelen = len(line)
        if linelen <= width:
            result.append(string)
            continue

        if snipat is None or snipat == 1:
            split = linelen
        elif 0 < abs(snipat) < 1:
            split = int(math.floor(linelen * snipat))
        else:
            split = int(snipat)

        if split < 0:
            split += linelen
            split = max(split, 0)

        if esize == 0 or width < esize + 1:
            if split <= 0:
                result.append(string[-width:])
            else:
                result.append(string[:width])
        elif split <= 0:
            result.append(ellipsis + string[esize - width :])
        elif split >= linelen or width < esize + 4:
            result.append(string[: width - esize] + ellipsis)
        else:
            splitlen = linelen - width + esize
            end1 = split - splitlen // 2
            end2 = end1 + splitlen
            result.append(string[:end1] + ellipsis + string[end2:])

    if isinstance(string, bytes):  # type: ignore[unreachable]
        return b'\n'.join(result)
    return '\n'.join(result)

def strptime(datetime_string: str, format: str | None = None, /) -> DateTime:
    """Return datetime corresponding to date string using common formats.

    Parameters:
        datetime_string:
            String representation of date and time.
        format:
            Format of `datetime_string`.
            By default, several datetime formats commonly found in TIFF files
            are parsed.

    Raises:
        ValueError: `datetime_string` does not match any format.

    Examples:
        >>> strptime('2022:08:01 22:23:24')
        datetime.datetime(2022, 8, 1, 22, 23, 24)

    """
    formats = {
        '%Y:%m:%d %H:%M:%S': 1,  # TIFF6 specification
        '%Y%m%d %H:%M:%S.%f': 2,  # MetaSeries
        '%Y-%m-%dT%H %M %S.%f': 3,  # Pilatus
        '%Y-%m-%dT%H:%M:%S.%f': 4,  # ISO
        '%Y-%m-%dT%H:%M:%S': 5,  # ISO, microsecond is 0
        '%Y:%m:%d %H:%M:%S.%f': 6,
        '%d/%m/%Y %H:%M:%S': 7,
        '%d/%m/%Y %H:%M:%S.%f': 8,
        '%m/%d/%Y %I:%M:%S %p': 9,
        '%m/%d/%Y %I:%M:%S.%f %p': 10,
        '%Y%m%d %H:%M:%S': 11,
        '%Y/%m/%d %H:%M:%S': 12,
        '%Y/%m/%d %H:%M:%S.%f': 13,
        '%Y-%m-%dT%H:%M:%S%z': 14,
        '%Y-%m-%dT%H:%M:%S.%f%z': 15,
    }
    if format is not None:
        formats[format] = 0  # highest priority; replaces existing key if any
    for format, _ in sorted(formats.items(), key=lambda item: item[1]):
        try:
            return DateTime.strptime(datetime_string, format)
        except ValueError:
            pass
    raise ValueError(
        f'time data {datetime_string!r} does not match any format'
    )

def stripnull(
    string: str | bytes,
    /,
    null: str | bytes | None = None,
    *,
    first: bool = True,
) -> str | bytes:
    r"""Return string truncated at first null character.

    Use to clean NULL terminated C strings.

    >>> stripnull(b'bytes\x00\x00')
    b'bytes'
    >>> stripnull(b'bytes\x00bytes\x00\x00', first=False)
    b'bytes\x00bytes'
    >>> stripnull('string\x00')
    'string'

    """
    if null is None:
        if isinstance(string, bytes):
            null = b'\x00'
        else:
            null = '\0'
    if first:
        i = string.find(null)  # type: ignore[arg-type]
        return string if i < 0 else string[:i]
    null = null[0]  # type: ignore[assignment]
    i = len(string)
    while i:
        i -= 1
        if string[i] != null:
            break
    else:
        i = -1
    return string[: i + 1]
