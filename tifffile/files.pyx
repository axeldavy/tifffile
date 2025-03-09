#cython: language_level=3
#cython: boundscheck=True
#cython: wraparound=True
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++


from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
import glob
import os
import re
import threading
import warnings

#from .series import TiledSequence
from .types import TIFF
from .utils import create_output, natural_sorted, snipstr
#from .zarr import ZarrFileSequenceStore

from cpython.buffer cimport PyBUF_SIMPLE, PyBuffer_FillInfo
from cpython.bytes cimport PyBytes_AsString
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy
from .utils cimport lock_gil_friendly, product, recursive_mutex, unique_lock
import numpy as np
#cimport numpy as np
cimport cython

#np.import_array()

def FILE_PATTERNS(self) -> dict[str, str]:
        # predefined FileSequence patterns
        return {
            'axes': r"""(?ix)
                # matches Olympus OIF and Leica TIFF series
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
                """
        }

def parse_filenames(
    files: Sequence[str],
    /,
    pattern: str | None = None,
    axesorder: Sequence[int] | None = None,
    categories: dict[str, dict[str, int]] | None = None,
    *,
    _shape: Sequence[int] | None = None,
) -> tuple[
    tuple[str, ...], tuple[int, ...], list[tuple[int, ...]], Sequence[str]
]:
    r"""Return shape and axes from sequence of file names matching pattern.

    Parameters:
        files:
            Sequence of file names to parse.
        pattern:
            Regular expression pattern matching axes names and chunk indices
            in file names.
            By default, no pattern matching is performed.
            Axes names can be specified by matching groups preceding the index
            groups in the file name, be provided as group names for the index
            groups, or be omitted.
            The predefined 'axes' pattern matches Olympus OIF and Leica TIFF
            series.
        axesorder:
            Indices of axes in pattern. By default, axes are returned in the
            order they appear in pattern.
        categories:
            Map of index group matches to integer indices.
            `{'axislabel': {'category': index}}`
        _shape:
            Shape of file sequence. The default is
            `maximum - minimum + 1` of the parsed indices for each dimension.

    Returns:
        - Axes names for each dimension.
        - Shape of file series.
        - Index of each file in shape.
        - Filtered sequence of file names.

    Examples:
        >>> parse_filenames(
        ...     ['c1001.ext', 'c2002.ext'], r'([^\d])(\d)(?P<t>\d+)\.ext'
        ... )
        (('c', 't'), (2, 2), [(0, 0), (1, 1)], ['c1001.ext', 'c2002.ext'])

    """
    # TODO: add option to filter files that do not match pattern

    shape = None if _shape is None else tuple(_shape)
    if pattern is None:
        if shape is not None and (len(shape) != 1 or shape[0] < len(files)):
            raise ValueError(
                f'shape {(len(files),)} does not fit provided shape {shape}'
            )
        return (
            ('I',),
            (len(files),),
            [(i,) for i in range(len(files))],
            files,
        )

    pattern = FILE_PATTERNS().get(pattern, pattern)
    if not pattern:
        raise ValueError('invalid pattern')
    if isinstance(pattern, str):
        pattern_compiled = re.compile(pattern)
    elif hasattr(pattern, 'groupindex'):
        pattern_compiled = pattern
    else:
        raise ValueError('invalid pattern')

    if categories is None:
        categories = {}

    def parse(fname: str, /) -> tuple[tuple[str, ...], tuple[int, ...]]:
        # return axes names and indices from file name
        assert categories is not None
        dims: list[str] = []
        indices: list[int] = []
        groupindex = {v: k for k, v in pattern_compiled.groupindex.items()}
        match = pattern_compiled.search(fname)
        if match is None:
            raise ValueError(f'pattern does not match file name {fname!r}')
        ax = None
        for i, m in enumerate(match.groups()):
            if m is None:
                continue
            if i + 1 in groupindex:
                ax = groupindex[i + 1]
            elif m[0].isalpha():
                ax = m  # axis label for next index
                continue
            if ax is None:
                ax = 'Q'  # no preceding axis letter
            try:
                if ax in categories:
                    m = categories[ax][m]
                m = int(m)
            except Exception as exc:
                raise ValueError(f'invalid index {m!r}') from exc
            indices.append(m)
            dims.append(ax)
            ax = None
        return tuple(dims), tuple(indices)

    normpaths = [os.path.normpath(f) for f in files]
    if len(normpaths) == 1:
        prefix_str = os.path.dirname(normpaths[0])
    else:
        prefix_str = os.path.commonpath(normpaths)
    prefix = len(prefix_str)

    dims: tuple[str, ...] | None = None
    indices: list[tuple[int, ...]] = []
    for fname in normpaths:
        lbl, idx = parse(fname[prefix:])
        if dims is None:
            dims = lbl
            if axesorder is not None and (
                len(axesorder) != len(dims)
                or any(i not in axesorder for i in range(len(dims)))
            ):
                raise ValueError(
                    f'invalid axesorder {axesorder!r} for {dims!r}'
                )
        elif dims != lbl:
            raise ValueError('dims do not match within image sequence')
        if axesorder is not None:
            idx = tuple(idx[i] for i in axesorder)
        indices.append(idx)

    assert dims is not None
    if axesorder is not None:
        dims = tuple(dims[i] for i in axesorder)

    # determine shape
    indices_array = np.array(indices, dtype=np.intp)
    parsedshape = np.max(indices, axis=0)

    if shape is None:
        startindex = np.min(indices_array, axis=0)
        indices_array -= startindex
        parsedshape -= startindex
        parsedshape += 1
        shape = tuple(int(i) for i in parsedshape.tolist())
    elif len(parsedshape) != len(shape) or any(
        i > j for i, j in zip(shape, parsedshape)
    ):
        raise ValueError(
            f'parsed shape {parsedshape} does not fit provided shape {shape}'
        )

    indices_list: list[list[int]]
    indices_list = indices_array.tolist()  # type: ignore[assignment]
    indices = [tuple(index) for index in indices_list]

    return dims, shape, indices, files

@cython.final
cdef class FileHandleLock:
    """
    A context manager that links to the FileHandler internal mutex
    """

    cdef FileHandle _filehandle
    cdef unique_lock[recursive_mutex] m

    def __cinit__(self, FileHandle filehandle):
        self._filehandle = filehandle

    def __enter__(self):
        assert not self.m.owns_lock()
        lock_gil_friendly(self.m, self._filehandle._mutex)

    def __exit__(self, exc_type, exc_value, traceback):
        assert self.m.owns_lock()
        self.m.unlock()

@cython.final
cdef class FileHandle:
    """Binary file handle.

    A limited, special purpose binary file handle that can:

    - handle embedded files (for example, LSM within LSM files).
    - re-open closed files (for multi-file formats, such as OME-TIFF).
    - read and write NumPy arrays and records from file-like objects.

    When initialized from another file handle, do not use the other handle
    unless this FileHandle is closed.

    FileHandle instances are not thread-safe.

    Parameters:
        file:
            File name or seekable binary stream, such as open file,
            BytesIO, or fsspec OpenFile.
        mode:
            File open mode if `file` is file name.
            The default is 'rb'. Files are always opened in binary mode.
        name:
            Name of file if `file` is binary stream.
        offset:
            Start position of embedded file.
            The default is the current file position.
        size:
            Size of embedded file.
            The default is the number of bytes from `offset` to
            the end of the file.
    """
    
    def __cinit__(self):
        self._read_cache = OrderedDict()
        self._max_cache_len = 10 # max 10 chunks in cache
        self._chunk_size = 4096 # 4 KB default chunk size for buffered reading

    def __init__(
        self,
        file,
        mode='rb',
        *,
        name='',
        offset=-1,
        size=-1,
    ):
        self._mode = mode
        self._fh = None
        self._file = file  # reference to original argument for re-opening
        self._name = name 
        self._dir = ''
        self._global_offset = offset
        self._size = size
        self._close = True
        self.open()
        assert self._fh is not None

    cpdef void open(self):
        """Open or re-open file."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        if self._fh is not None:
            return  # file is open

        if isinstance(self._file, os.PathLike):
            self._file = os.fspath(self._file)

        if isinstance(self._file, str):
            # file name
            if self._mode[-1:] != 'b':
                self._mode += 'b'
            if self._mode not in {'rb', 'r+b', 'wb', 'xb'}:
                raise ValueError(f'invalid mode {self._mode}')
            self._file = os.path.realpath(self._file)
            self._dir, self._name = os.path.split(self._file)
            self._fh = open(self._file, self._mode, encoding=None)
            self._close = True
            self._global_offset = max(0, self._global_offset)
        elif isinstance(self._file, FileHandle):
            # FileHandle
            self._fh = self._file._fh
            self._global_offset = max(0, self._global_offset)
            self._global_offset += self._file._offset
            self._close = False
            if not self._name:
                if self._global_offset:
                    name, ext = os.path.splitext(self._file._name)
                    self._name = f'{name}@{self._global_offset}{ext}'
                else:
                    self._name = self._file._name
            self._mode = self._file._mode
            self._dir = self._file._dir
        elif hasattr(self._file, 'seek'):
            # binary stream: open file, BytesIO, fsspec LocalFileOpener
            self._fh = self._file
            try:
                self._fh.tell()
            except Exception as exc:
                raise ValueError('binary stream is not seekable') from exc

            if self._global_offset < 0:
                self._global_offset = self._fh.tell()
            self._close = False
            if not self._name:
                try:
                    self._dir, self._name = os.path.split(self._fh.name)
                except AttributeError:
                    try:
                        self._dir, self._name = os.path.split(self._fh.path)
                    except AttributeError:
                        self._name = 'Unnamed binary stream'
            try:
                self._mode = self._fh.mode
            except AttributeError:
                pass
        elif hasattr(self._file, 'open'):
            # fsspec OpenFile
            _file = self._file
            self._fh = _file.open()
            try:
                self._fh.tell()
            except Exception as exc:
                try:
                    self._fh.close()
                except Exception:
                    pass
                raise ValueError('OpenFile is not seekable') from exc

            if self._global_offset < 0:
                self._global_offset = self._fh.tell()
            self._close = True
            if not self._name:
                try:
                    self._dir, self._name = os.path.split(_file.path)
                except AttributeError:
                    self._name = 'Unnamed binary stream'
            try:
                self._mode = _file.mode
            except AttributeError:
                pass

        else:
            raise ValueError(
                'the first parameter must be a file name '
                'or seekable binary file object, '
                f'not {type(self._file)!r}'
            )

        assert self._fh is not None

        if self._global_offset:
            self._fh.seek(self._global_offset)

        if self._size < 0:
            pos = self._fh.tell()
            self._fh.seek(self._global_offset, os.SEEK_END)
            self._size = self._fh.tell()
            self._fh.seek(pos)

    cpdef void close(self):
        """Close file handle."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        if self._close and self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                # PermissionError on MacOS. See issue #184
                pass
            self._fh = None

    cpdef int64_t fileno(self):
        """Return underlying file descriptor if exists, else raise OSError."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        assert self._fh is not None
        try:
            return self._fh.fileno()
        except (OSError, AttributeError) as exc:
            raise OSError(
                f'{type(self._fh)} does not have a file descriptor'
            ) from exc

    cpdef bint writable(self):
        """Return True if stream supports writing."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        assert self._fh is not None
        if hasattr(self._fh, 'writable'):
            return self._fh.writable()
        return False

    cpdef bint seekable(self):
        """Return True if stream supports random access."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        return True

    cpdef int64_t tell(self):
        """Return file's current position."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        assert self._fh is not None
        return self._current_offset - self._global_offset

    cdef int64_t _seek(self, int64_t offset, int64_t whence):
        """Actually seek the underlying file"""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        if offset == self._actual_position and whence == 0:
            return self._actual_position
        self._actual_position = self._fh.seek(offset, whence)
        return self._actual_position

    cdef bytes _actual_read(self, int64_t offset, int64_t size):
        """Read with no caching"""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        self._seek(offset, 0)
        if offset + size > self._size:
            size = self.size - offset
        if size <= 0:
            return b""
        self._actual_position += size
        return self._fh.read(size)

    cdef bytes _read(self, int64_t offset, int64_t size):
        """Read and cache (if read is not too large) the requested data"""
        # Do not use cache for large reads
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        if size > self._chunk_size:
            return self._actual_read(offset, size)

        cdef int64_t chunk_id_start = offset // self._chunk_size
        cdef int64_t chunk_id_end = (offset+size) // self._chunk_size
        assert (chunk_id_start == chunk_id_end
                or chunk_id_start == chunk_id_end - 1)

        cdef bytes chunk1, chunk2
        # load the chunks in cache if needed
        if chunk_id_start not in self._read_cache:
            chunk1 = self._actual_read(
                chunk_id_start * self._chunk_size,
                self._chunk_size
            )
            self._read_cache[chunk_id_start] = chunk1
        else:
            chunk1 = self._read_cache[chunk_id_start]
            self._read_cache.move_to_end(chunk_id_start)

        if (chunk_id_start != chunk_id_end and
            chunk_id_end not in self._read_cache):
            chunk2 = self._actual_read(
                chunk_id_end * self._chunk_size,
                self._chunk_size
            )
            self._read_cache[chunk_id_end] = chunk2
        else:
            chunk2 = self._read_cache[chunk_id_end]
            self._read_cache.move_to_end(chunk_id_end)

        # Delete old cache entries
        while len(self._read_cache) > self._max_cache_len:
            self._read_cache.popitem(last=False)

        # Concatenate the chunks
        if chunk_id_start == chunk_id_end:
            return chunk1[offset % self._chunk_size:offset % self._chunk_size+size]
        else:
            return (chunk1[offset % self._chunk_size:] +
                    chunk2[:size - (offset % self._chunk_size)])


    cpdef int64_t seek(self, int64_t offset, int64_t whence=0):
        """Set file's current position.

        Parameters:
            offset:
                Position of file handle relative to position indicated
                by `whence`.
            whence:
                Relative position of `offset`.
                0 (`os.SEEK_SET`) beginning of file (default).
                1 (`os.SEEK_CUR`) current position.
                2 (`os.SEEK_END`) end of file.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        assert self._fh is not None
        cdef int64_t target_offset = offset
        if self._global_offset > 0:
            if whence == 0:
                target_offset += self._global_offset
            elif whence == 2:
                target_offset += self._global_offset + self._size
        if whence == 0 and target_offset < 0:
            target_offset = 0
        elif whence == 2 and target_offset > self._size:
            target_offset = self._size
        self._current_offset = target_offset
        return target_offset - self._global_offset

    cpdef bytes read(self, int64_t size=-1):
        """Return bytes read from file.

        Parameters:
            size:
                Number of bytes to read from file.
                By default, read until the end of the file.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        if size < 0 and self._global_offset > 0:
            size = self._size
        assert self._fh is not None
        cdef int64_t current_pos = self.tell()
        cdef bytes result = self._read(self._current_offset, size)
        self.seek(current_pos+len(result), 0)
        return result

    cpdef bytes read_at(self, int64_t offset, int64_t size):
        """Return bytes read from file at offset.

        It is equivalent to
        self.seek(offset)
        self.read(size)

        except that the file lock is held for the
        entire operation.

        Parameters:
            offset:
                Position in file to start reading from.
            size:
                Number of bytes to read from file.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        assert self._fh is not None
        self.seek(offset, 0)
        cdef bytes result = self._read(self._current_offset, size)
        self.seek(offset+len(result), 0)
        return result

    cpdef int64_t readinto(self, buffer):
        """Read bytes from file into buffer.

        Parameters:
            buffer: Buffer to read into.

        Returns:
            Number of bytes read from file.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        assert self._fh is not None
        cdef bytes content = self._read(self._current_offset, len(buffer))
        cdef int64_t n = len(content)
        cdef char[::1] dst = buffer
        memcpy(&dst[0], PyBytes_AsString(content), n)
        return n

    cpdef int64_t write(self, bytes buffer):
        """Write bytes to file and return number of bytes written.

        Parameters:
            buffer: Bytes to write to file.

        Returns:
            Number of bytes written.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        assert self._fh is not None
        # invalidate cache
        self._read_cache.clear()
        # resolve position
        self._seek(self._current_offset, 0)
        if self.tell() + len(buffer) > self._size:
            self._size = self.tell() + len(buffer)
        # perform write
        return self._fh.write(buffer)

    cpdef void flush(self):
        """Flush write buffers of stream if applicable."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        assert self._fh is not None
        if hasattr(self._fh, 'flush'):
            self._fh.flush()

    cpdef object memmap_array(
        self,
        object dtype,
        tuple shape,
        int64_t offset=0,
        str mode='r',
        str order='C',
    ):
        """Return `numpy.memmap` of array data stored in file.

        Parameters:
            dtype:
                Data type of array in file.
            shape:
                Shape of array in file.
            offset:
                Start position of array-data in file.
            mode:
                File is opened in this mode. The default is read-only.
            order:
                Order of ndarray memory layout. The default is 'C'.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        if not self.is_file:
            raise ValueError('cannot memory-map file without fileno')
        assert self._fh is not None
        return np.memmap(
            self._fh,  # type: ignore[call-overload]
            dtype=dtype,
            mode=mode,
            offset=self._global_offset + offset,
            shape=shape,
            order=order,
        )

    cpdef object read_array(
        self,
        dtype,
        int64_t count=-1,
        int64_t offset=0,
        out=None,
    ):
        """Return NumPy array from file in native byte order.

        Parameters:
            dtype:
                Data type of array to read.
            count:
                Number of items to read. By default, all items are read.
            offset:
                Start position of array-data in file.
            out:
                NumPy array to read into. By default, a new array is created.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        dtype = np.dtype(dtype)
        cdef int64_t n, nbytes

        if count < 0:
            nbytes = self._size if out is None else out.nbytes
            count = nbytes // dtype.itemsize
        else:
            nbytes = count * dtype.itemsize

        result = np.empty(count, dtype) if out is None else out

        if result.nbytes != nbytes:
            raise ValueError('size mismatch')

        assert self._fh is not None

        self.seek(offset, 0)

        try:
            n = self.readinto(result)
        except AttributeError:
            result[:] = np.frombuffer(self.read(nbytes), dtype).reshape(
                result.shape
            )
            n = nbytes

        if n != nbytes:
            raise ValueError(f'failed to read {nbytes} bytes, got {n}')

        if not result.dtype.isnative:
            if not dtype.isnative:
                result.byteswap(True)
            result = result.view(result.dtype.newbyteorder())
        elif result.dtype.isnative != dtype.isnative:
            result.byteswap(True)

        if out is not None:
            if hasattr(out, 'flush'):
                out.flush()

        return result

    cpdef object read_record(
        self,
        dtype,
        shape=1,
        byteorder=None,
    ):
        """Return NumPy record from file.

        Parameters:
            dtype:
                Data type of record array to read.
            shape:
                Shape of record array to read.
            byteorder:
                Byte order of record array to read.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        assert self._fh is not None

        dtype = np.dtype(dtype)
        if byteorder is not None:
            dtype = dtype.newbyteorder(byteorder)

        try:
            record = np.rec.fromfile(  # type: ignore[call-overload]
                self._fh, dtype, shape
            )
        except Exception:
            if shape is None:
                shape = self._size // dtype.itemsize
            size = product(shape) * dtype.itemsize
            # data = bytearray(size)
            # n = self._fh.readinto(data)
            # data = data[:n]
            # TODO: record is not writable
            data = self.read(size)
            record = np.rec.fromstring(
                data,
                dtype,
                shape,
            )
        return record[0] if shape == 1 else record

    cpdef int64_t write_empty(self, int64_t size):
        """Append null-bytes to file.

        Parameters:
            size: Number of null-bytes to write to file.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        if size < 1:
            return 0
        assert self._fh is not None
        self.seek(self._size, 0)
        self.write(b'\x00' * size)
        return size

    cpdef int64_t write_array(
        self,
        data,
        dtype=None,
    ):
        """Write NumPy array to file in C contiguous order.

        Parameters:
            data: Array to write to file.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        assert self._fh is not None
        pos = self.tell()
        # writing non-contiguous arrays is very slow
        data = np.ascontiguousarray(data, dtype)
        try:
            data.tofile(self._fh)
        except io.UnsupportedOperation:
            # numpy cannot write to BytesIO
            self.write(data.tobytes())
        return self.tell() - pos

    def read_segments(
        self,
        offsets,
        bytecounts,
        indices=None,
        bint sort=True,
        int64_t buffersize=-1,
        bint flat=True,
    ):
        """Return iterator over segments read from file and their indices.

        The purpose of this function is to

        - reduce small or random reads.
        - reduce acquiring reentrant locks.
        - synchronize seeks and reads.
        - limit size of segments read into memory at once.
          (ThreadPoolExecutor.map is not collecting iterables lazily).

        Parameters:
            offsets:
                Offsets of segments to read from file.
            bytecounts:
                Byte counts of segments to read from file.
            indices:
                Indices of segments in image.
                The default is `range(len(offsets))`.
            sort:
                Read segments from file in order of their offsets.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.
            flat:
                If *True*, return iterator over individual (segment, index)
                tuples.
                Else, return an iterator over a list of (segment, index)
                tuples that were acquired in one pass.

        Yields:
            Individual or lists of `(segment, index)` tuples.

        """
        assert self._fh is not None
        cdef int64_t i, j, offset, bytecount, length, index
        cdef int64_t size, start, stop

        length = len(offsets)
        if length < 1:
            return
        if length == 1:
            index = 0 if indices is None else indices[0]
            if bytecounts[index] > 0 and offsets[index] > 0:
                data = self.read_at(offsets[index], bytecounts[index])
            else:
                data = None
            yield (data, index) if flat else [(data, index)]
            return

        if buffersize < 0:
            buffersize = TIFF.BUFFERSIZE

        if indices is None:
            segments = [(i, offsets[i], bytecounts[i]) for i in range(length)]
        else:
            segments = [
                (indices[i], offsets[i], bytecounts[i]) for i in range(length)
            ]
        if sort:
            segments = sorted(segments, key=lambda x: x[1])

        iscontig = True
        for i in range(length - 1):
            _, offset, bytecount = segments[i]
            nextoffset = segments[i + 1][1]
            if offset == 0 or bytecount == 0 or nextoffset == 0:
                continue
            if offset + bytecount != nextoffset:
                iscontig = False
                break

        result = []

        if iscontig:
            # consolidate reads
            i = 0
            while i < length:
                j = i
                offset = -1
                bytecount = 0
                while bytecount <= buffersize and i < length:
                    _, o, b = segments[i]
                    if o > 0 and b > 0:
                        if offset < 0:
                            offset = o
                        bytecount += b
                    i += 1

                if offset < 0:
                    data = None
                else:
                    data = self.read_at(offset, bytecount)
                start = 0
                stop = 0
                result = []
                while j < i:
                    index, offset, bytecount = segments[j]
                    if offset > 0 and bytecount > 0:
                        stop += bytecount
                        result.append(
                            (data[start:stop], index)  # type: ignore[index]
                        )
                        start = stop
                    else:
                        result.append((None, index))
                    j += 1
                if flat:
                    yield from result
                else:
                    yield result
            return

        i = 0
        cdef unique_lock[recursive_mutex] m
        while i < length:
            result = []
            size = 0
            lock_gil_friendly(m, self._mutex)
            while size <= buffersize and i < length:
                index, offset, bytecount = segments[i]
                if offset > 0 and bytecount > 0:
                    self.seek(offset)
                    result.append((self.read(bytecount), index))
                    # buffer = bytearray(bytecount)
                    # n = fh.readinto(buffer)
                    # data.append(buffer[:n])
                    size += bytecount
                else:
                    result.append((None, index))
                i += 1
            m.unlock()
            if flat:
                yield from result
            else:
                yield result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self._file = None

    def __repr__(self):
        return f'<tifffile.FileHandle {snipstr(self._name, 32)!r}>'

    def __str__(self):
        return '\n '.join(
            (
                'FileHandle',
                self._name,
                self._dir,
                f'{self._size} bytes',
                'closed' if self._fh is None else 'open',
            )
        )

    @property
    def name(self):
        """Name of file or stream."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        return self._name

    @property
    def dirname(self):
        """Directory in which file is stored."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        return self._dir

    @property
    def path(self):
        """Absolute path of file."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        return os.path.join(self._dir, self._name)

    @property
    def extension(self):
        """File name extension of file or stream."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        name, ext = os.path.splitext(self._name.lower())
        if ext and name.endswith('.ome'):
            ext = '.ome' + ext
        return ext

    @property
    def size(self):
        """Size of file in bytes."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        return self._size

    @property
    def closed(self):
        """File is closed."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        return self._fh is None

    @property
    def is_file(self):
        """File has fileno and can be memory-mapped."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        try:
            self._fh.fileno()  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    @property
    def lock(self):
        return FileHandleLock(self)

    def set_lock(bint v):
        return


@cython.final
cdef class FileCache:
    """Keep FileHandles open.

    Parameters:
        size: Maximum number of files to keep open. The default is 8.
        lock: Reentrant lock to synchronize reads and writes.

    """

    cdef int64_t size
    """Maximum number of files to keep open."""

    cdef dict files
    """Reference counts of opened files."""

    cdef set keep
    """Set of files to keep open."""

    cdef list past
    """FIFO list of opened files."""

    cdef recursive_mutex _mutex

    def __cinit__(self):
        self.past = []
        self.files = {}
        self.keep = set()
        self.size = 8

    def __init__(
        self,
        int64_t size=-1
    ):
        if size >= 0:
            self.size = size

    cpdef void open(self, FileHandle fh):
        """Open file, re-open if necessary."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        if fh in self.files:
            self.files[fh] += 1
        elif fh.closed:
            fh.open()
            self.files[fh] = 1
            self.past.append(fh)
        else:
            self.files[fh] = 2
            self.keep.add(fh)
            self.past.append(fh)

    cpdef void close(self, FileHandle fh):
        """Close least recently used open files."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        if fh in self.files:
            self.files[fh] -= 1
        self._trim()

    cpdef void clear(self):
        """Close all opened files if not in use when opened first."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        for fh, refcount in list(self.files.items()):
            if fh not in self.keep:
                fh.close()
                del self.files[fh]
                del self.past[self.past.index(fh)]

    cpdef bytes read(
        self,
        FileHandle fh,
        int64_t offset,
        int64_t bytecount,
        int64_t whence=0,
    ):
        """Return bytes read from binary file.

        Parameters:
            fh:
                File handle to read from.
            offset:
                Position in file to start reading from relative to the
                position indicated by `whence`.
            bytecount:
                Number of bytes to read.
            whence:
                Relative position of offset.
                0 (`os.SEEK_SET`) beginning of file (default).
                1 (`os.SEEK_CUR`) current position.
                2 (`os.SEEK_END`) end of file.

        """
        # this function is more efficient than
        # filecache.open(fh)
        # with lock:
        #     fh.seek()
        #     data = fh.read()
        # filecache.close(fh)
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        b = fh not in self.files
        if b:
            if fh.closed:
                fh.open()
                self.files[fh] = 0
            else:
                self.files[fh] = 1
                self.keep.add(fh)
            self.past.append(fh)
        fh.seek(offset, whence)
        data = fh.read(bytecount)
        if b:
            self._trim()
        return data

    cpdef int64_t write(
        self,
        FileHandle fh,
        int64_t offset,
        bytes data,
        int64_t whence=0,
    ):
        """Write bytes to binary file.

        Parameters:
            fh:
                File handle to write to.
            offset:
                Position in file to start writing from relative to the
                position indicated by `whence`.
            value:
                Bytes to write.
            whence:
                Relative position of offset.
                0 (`os.SEEK_SET`) beginning of file (default).
                1 (`os.SEEK_CUR`) current position.
                2 (`os.SEEK_END`) end of file.

        """
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        b = fh not in self.files
        if b:
            if fh.closed:
                fh.open()
                self.files[fh] = 0
            else:
                self.files[fh] = 1
                self.keep.add(fh)
            self.past.append(fh)
        fh.seek(offset, whence)
        written = fh.write(data)
        if b:
            self._trim()
        return written

    cpdef void _trim(self):
        """Trim file cache."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        index = 0
        size = len(self.past)
        while index < size > self.size:
            fh = self.past[index]
            if fh not in self.keep and self.files[fh] <= 0:
                fh.close()
                del self.files[fh]
                del self.past[index]
                size -= 1
            else:
                index += 1

    def __len__(self):
        """Return number of open files."""
        cdef unique_lock[recursive_mutex] m
        lock_gil_friendly(m, self._mutex)
        return len(self.files)

    def __repr__(self):
        return f'<tifffile.FileCache @0x{id(self):016X}>'


cdef class FileSequence:
    r"""Sequence of files containing compatible array data.

    Parameters:
        imread:
            Function to read image array from single file.
        files:
            Glob filename pattern or sequence of file names.
            If *None*, use '\*'.
            All files must contain array data of same shape and dtype.
            Binary streams are not supported.
        container:
            Name or open instance of ZIP file in which files are stored.
        sort:
            Function to sort file names if `files` is a pattern.
            The default is :py:func:`natural_sorted`.
            If *False*, disable sorting.
        parse:
            Function to parse sequence of sorted file names to dims, shape,
            chunk indices, and filtered file names.
            The default is :py:func:`parse_filenames` if `kwargs`
            contains `'pattern'`.
        **kwargs:
            Additional arguments passed to `parse` function.

    Examples:
        >>> filenames = ['temp_C001T002.tif', 'temp_C001T001.tif']
        >>> ims = TiffSequence(filenames, pattern=r'_(C)(\d+)(T)(\d+)')
        >>> ims[0]
        'temp_C001T002.tif'
        >>> ims.shape
        (1, 2)
        >>> ims.axes
        'CT'

    """

    cdef object imread
    """Function to read image array from single file."""

    cdef tuple shape
    """Shape of file series. Excludes shape of chunks in files."""

    cdef str axes
    """Character codes for dimensions in shape."""

    cdef tuple dims
    """Names of dimensions in shape."""

    cdef tuple indices
    """Indices of files in shape."""

    cdef list _files  # list of file names
    cdef object _container  # TODO: container type?

    def __cinit__(self):
        self._files = []
        self._container = None

    def __init__(
        self,
        imread,
        files=None,
        container=None,
        sort=None,
        parse=None,
        **kwargs
    ):
        sort_func = None

        if files is None:
            files = '*'
        if sort is None:
            sort_func = natural_sorted
        elif callable(sort):
            sort_func = sort
        elif sort:
            sort_func = natural_sorted
        # elif not sort:
        #     sort_func = None

        self._container = container
        if container is not None:
            import fnmatch

            if isinstance(container, (str, os.PathLike)):
                import zipfile

                self._container = zipfile.ZipFile(container)
            elif not hasattr(self._container, 'open'):
                raise ValueError('invalid container')
            if isinstance(files, str):
                files = fnmatch.filter(self._container.namelist(), files)
                if sort_func is not None:
                    files = sort_func(files)
        elif isinstance(files, os.PathLike):
            files = [os.fspath(files)]
            if sort is not None and sort_func is not None:
                files = sort_func(files)
        elif isinstance(files, str):
            files = glob.glob(files)
            if sort_func is not None:
                files = sort_func(files)

        files = [os.fspath(f) for f in files]  # type: ignore[union-attr]
        if not files:
            raise ValueError('no files found')

        if not callable(imread):
            raise ValueError('invalid imread function')

        if container:
            # redefine imread to read from container
            def imread_(
                fname, _imread=imread, **kwargs
            ):
                with self._container.open(fname) as handle1:
                    with io.BytesIO(handle1.read()) as handle2:
                        return _imread(handle2, **kwargs)

            imread = imread_

        if parse is None and kwargs.get('pattern', None):
            parse = parse_filenames

        if parse:
            try:
                dims, shape, indices, files = parse(files, **kwargs)
            except ValueError as exc:
                raise ValueError('failed to parse file names') from exc
        else:
            dims = ('sequence',)
            shape = (len(files),)
            indices = tuple((i,) for i in range(len(files)))

        assert isinstance(files, list) and isinstance(files[0], str)
        codes = TIFF.AXES_CODES
        axes = ''.join(codes.get(dim.lower(), dim[0].upper()) for dim in dims)

        self._files = files
        self.imread = imread
        self.axes = axes
        self.dims = tuple(dims)
        self.shape = tuple(shape)
        self.indices = indices

    def asarray(
        self,
        imreadargs=None,
        chunkshape=None,
        chunkdtype=None,
        axestiled=None,
        out_inplace=None,
        int64_t ioworkers=1,
        out=None,
        **kwargs
    ):
        """Return images from files as NumPy array.

        Parameters:
            imreadargs:
                Arguments passed to :py:attr:`FileSequence.imread`.
            chunkshape:
                Shape of chunk in each file.
                Must match ``FileSequence.imread(file, **imreadargs).shape``.
                By default, this is determined by reading the first file.
            chunkdtype:
                Data type of chunk in each file.
                Must match ``FileSequence.imread(file, **imreadargs).dtype``.
                By default, this is determined by reading the first file.
            axestiled:
                Axes to be tiled.
                Map stacked sequence axis to chunk axis.
            ioworkers:
                Maximum number of threads to execute
                :py:attr:`FileSequence.imread` asynchronously.
                If *0*, use up to :py:attr:`_TIFF.MAXIOWORKERS` threads.
                Using threads can significantly improve runtime when reading
                many small files from a network share.
            out_inplace:
                :py:attr:`FileSequence.imread` decodes directly to the output
                instead of returning an array, which is copied to the output.
                Not all imread functions support this, especially in
                non-contiguous cases.
            out:
                Specifies how image array is returned.
                By default, create a new array.
                If a *numpy.ndarray*, a writable array to which the images
                are copied.
                If *'memmap'*, create a memory-mapped array in a temporary
                file.
                If a *string* or *open file*, the file used to create a
                memory-mapped array.
            **kwargs:
                Arguments passed to :py:attr:`FileSequence.imread` in
                addition to `imreadargs`.

        Raises:
            IndexError, ValueError: Array shapes do not match.

        """
        from .tifffile import TiledSequence
        # TODO: deprecate kwargs?
        files = self._files
        if imreadargs is not None:
            kwargs |= imreadargs

        if ioworkers is None or ioworkers < 1:
            ioworkers = TIFF.MAXIOWORKERS
        ioworkers = min(len(files), ioworkers)
        assert isinstance(ioworkers, int)  # mypy bug?

        if out_inplace is None: # TODO and self.imread == imread:
            out_inplace = True
        else:
            out_inplace = bool(out_inplace)

        if chunkshape is None or chunkdtype is None:
            im = self.imread(files[0], **kwargs)
            chunkshape = im.shape
            chunkdtype = im.dtype
            del im
        chunkdtype = np.dtype(chunkdtype)
        assert chunkshape is not None

        if axestiled:
            tiled = TiledSequence(self.shape, chunkshape, axestiled=axestiled)
            result = create_output(out, tiled.shape, chunkdtype)

            def func(index, fname):
                # read single image from file into result
                # if index is None:
                #     return
                if out_inplace:
                    self.imread(fname, out=result[index], **kwargs)
                else:
                    im = self.imread(fname, **kwargs)
                    result[index] = im
                    del im  # delete memory-mapped file

            if ioworkers < 2:
                for index, fname in zip(tiled.slices(self.indices), files):
                    func(index, fname)
            else:
                with ThreadPoolExecutor(ioworkers) as executor:
                    for _ in executor.map(
                        func, tiled.slices(self.indices), files
                    ):
                        pass
        else:
            shape = self.shape + chunkshape
            result = create_output(out, shape, chunkdtype)
            result = result.reshape(-1, *chunkshape)

            def func(index, fname):
                # read single image from file into result
                if index is None:
                    return
                index_ = int(
                    np.ravel_multi_index(
                        index,  # type: ignore[arg-type]
                        self.shape,
                    )
                )
                if out_inplace:
                    self.imread(fname, out=result[index_], **kwargs)
                else:
                    im = self.imread(fname, **kwargs)
                    result[index_] = im
                    del im  # delete memory-mapped file

            if ioworkers < 2:
                for index, fname in zip(self.indices, files):
                    func(index, fname)
            else:
                with ThreadPoolExecutor(ioworkers) as executor:
                    for _ in executor.map(func, self.indices, files):
                        pass

            result.shape = shape

        return result

    def aszarr(self, **kwargs):
        """Return images from files as Zarr 2 store.

        Parameters:
            **kwargs: Arguments passed to :py:class:`ZarrFileSequenceStore`.

        """
        from .tifffile import ZarrFileSequenceStore
        return ZarrFileSequenceStore(self, **kwargs)

    cpdef void close(self):
        """Close open files."""
        if self._container is not None:
            self._container.close()
        self._container = None

    cpdef str commonpath(self):
        """Return longest common sub-path of each file in sequence."""
        if len(self._files) == 1:
            commonpath = os.path.dirname(self._files[0])
        else:
            commonpath = os.path.commonpath(self._files)
        return commonpath

    @property
    def files(self):
        """Deprecated. Use the FileSequence sequence interface.

        :meta private:

        """
        warnings.warn(
            '<tifffile.FileSequence.files> is deprecated since 2024.5.22. '
            'Use the FileSequence sequence interface.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self._files

    @property
    def files_missing(self):
        """Number of empty chunks."""
        return product(self.shape) - len(self._files)

    def __iter__(self):
        """Return iterator over all file names."""
        return iter(self._files)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, key):
        return self._files[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        return f'<tifffile.FileSequence @0x{id(self):016X}>'

    def __str__(self):
        file = str(self._container) if self._container else self._files[0]
        file = os.path.split(file)[-1]
        return '\n '.join(
            (
                self.__class__.__name__,
                file,
                f'files: {len(self._files)} ({self.files_missing} missing)',
                'shape: {}'.format(', '.join(str(i) for i in self.shape)),
                'dims: {}'.format(', '.join(s for s in self.dims)),
                # f'axes: {self.axes}',
            )
        )


@cython.final
cdef class TiffSequence(FileSequence):
    r"""Sequence of TIFF files containing compatible array data.

    Same as :py:class:`FileSequence` with the :py:func:`imread` function,
    `'\*.tif'` glob pattern, and `out_inplace` enabled by default.

    """

    def __init__(
        self,
        files=None,
        imread=None,
        **kwargs
    ):
        if imread is None:
            from. import tifffile
            imread = tifffile.imread
        super().__init__(imread, '*.tif' if files is None else files, **kwargs)

    def __repr__(self):
        return f'<tifffile.TiffSequence @0x{id(self):016X}>'
