# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import os
import io
import threading
import warnings
from collection import OrderedDict

from typing import Any, IO, Literal, Sequence, cast
from cpython.buffer cimport PyBUF_SIMPLE, PyBuffer_FillInfo
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy
import numpy as np
cimport numpy as np

np.import_array()

# Define the NullContext class
class NullContext:
    def __enter__(self):
        return None
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass


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
        self._buffer = NULL
        self._read_cache = OrderedDict()
        self._max_cache_len = 10 # max 10 chunks in cache
        self._chunk_size = 4096 # 4 KB default chunk size for buffered reading

    def __init__(
        self,
        file,
        mode=None,
        *,
        name=None,
        offset=None,
        size=None,
    ):
        self._mode = 'rb' if mode is None else mode
        self._fh = None
        self._file = file  # reference to original argument for re-opening
        self._name = name if name else ''
        self._dir = ''
        self._offset = -1 if offset is None else offset
        self._size = -1 if size is None else size
        self._close = True
        self._lock = NullContext()
        self.open()
        assert self._fh is not None

    cpdef void open(self) except *:
        """Open or re-open file."""
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
            self._offset = max(0, self._offset)
        elif isinstance(self._file, FileHandle):
            # FileHandle
            self._fh = self._file._fh
            self._offset = max(0, self._offset)
            self._offset += self._file._offset
            self._close = False
            if not self._name:
                if self._offset:
                    name, ext = os.path.splitext(self._file._name)
                    self._name = f'{name}@{self._offset}{ext}'
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

            if self._offset < 0:
                self._offset = self._fh.tell()
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

            if self._offset < 0:
                self._offset = self._fh.tell()
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

        if self._offset:
            self._fh.seek(self._offset)

        if self._size < 0:
            pos = self._fh.tell()
            self._fh.seek(self._offset, os.SEEK_END)
            self._size = self._fh.tell()
            self._fh.seek(pos)

    cpdef void close(self) except *:
        """Close file handle."""
        if self._close and self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                # PermissionError on MacOS. See issue #184
                pass
            self._fh = None

    cpdef int fileno(self) except *:
        """Return underlying file descriptor if exists, else raise OSError."""
        assert self._fh is not None
        try:
            return self._fh.fileno()
        except (OSError, AttributeError) as exc:
            raise OSError(
                f'{type(self._fh)} does not have a file descriptor'
            ) from exc

    cpdef bint writable(self) except *:
        """Return True if stream supports writing."""
        assert self._fh is not None
        if hasattr(self._fh, 'writable'):
            return self._fh.writable()
        return False

    cpdef bint seekable(self) except *:
        """Return True if stream supports random access."""
        return True

    cpdef int tell(self) except *:
        """Return file's current position."""
        assert self._fh is not None
        return self._fh.tell() - self._offset

    cdef bytes _read(self, int64_t offset, int64_t size):
        """Read and cache (if read is not too large) the requested data"""
        # Do not use cache for large reads
        if size > self._chunk_size:
            self._fh.seek(offset, 0)
            return self._fh.read(size)

        cdef int chunk_id_start = offset // self._chunk_size
        cdef int chunk_id_end = (offset+size) // self.chunk_size
        assert (chunk_id_start == chunk_id_end
                or chunk_id_start == chunk_id_end - 1)

        cdef bytes chunk1, chunk2
        cdef bint chunk1_loaded = False
        # load the chunks in cache if needed
        if chunk_id_start not in self._read_cache:
            self._fh.seek(chunk_id_start * self.chunk_size, 0)
            chunk1 = self._fh.read(self.chunk_size)
            self._read_cache[chunk_id_start] = chunk1
            chunk1_loaded = True
        else:
            chunk1 = self._read_cache[chunk_id_start]
            self._read_cache.move_to_end(chunk_id_start)

        if (chunk_id_start != chunk_id_end and
            chunk_id_end not in self._read_cache):
            if not chunk1_loaded:
                self._fh.seek(chunk_id_end * self.chunk_size, 0)
            chunk2 = self._fh.read(self.chunk_size)
            self._read_cache[chunk_id_end] = chunk2
        else:
            chunk2 = self._read_cache[chunk_id_end]
            self._read_cache.move_to_end(chunk_id_end)

        # Delete old cache entries
        while len(self._read_cache) > self._max_cache_len:
            self._read_cache.popitem(last=False)

        # Concatenate the chunks
        if chunk_id_start == chunk_id_end:
            return chunk1[offset % self.chunk_size:offset % self.chunk_size+size]
        else:
            return (chunk1[offset % self.chunk_size:] +
                    chunk2[:size - (offset % self.chunk_size)])


    cpdef int seek(self, int offset, int whence=0) except *:
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
        assert self._fh is not None
        if self._offset:
            if whence == 0:
                return (
                    self._fh.seek(self._offset + offset, whence) - self._offset
                )
            if whence == 2 and self._size > 0:
                return (
                    self._fh.seek(self._offset + self._size + offset, 0)
                    - self._offset
                )
        return self._fh.seek(offset, whence)

    cpdef bytes read(self, int size=-1) except *:
        """Return bytes read from file.

        Parameters:
            size:
                Number of bytes to read from file.
                By default, read until the end of the file.

        """
        if size < 0 and self._offset:
            size = self._size
        assert self._fh is not None
        return self._fh.read(size)

    cpdef int readinto(self, bytes buffer) except *:
        """Read bytes from file into buffer.

        Parameters:
            buffer: Buffer to read into.

        Returns:
            Number of bytes read from file.

        """
        assert self._fh is not None
        return self._fh.readinto(buffer)  # type: ignore[attr-defined]

    cpdef int write(self, bytes buffer) except *:
        """Write bytes to file and return number of bytes written.

        Parameters:
            buffer: Bytes to write to file.

        Returns:
            Number of bytes written.

        """
        assert self._fh is not None
        return self._fh.write(buffer)

    cpdef void flush(self) except *:
        """Flush write buffers of stream if applicable."""
        assert self._fh is not None
        if hasattr(self._fh, 'flush'):
            self._fh.flush()

    cpdef np.ndarray memmap_array(
        self,
        dtype,
        shape,
        int offset=0,
        str mode='r',
        str order='C',
    ) except *:
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
        if not self.is_file:
            raise ValueError('cannot memory-map file without fileno')
        assert self._fh is not None
        return np.memmap(
            self._fh,  # type: ignore[call-overload]
            dtype=dtype,
            mode=mode,
            offset=self._offset + offset,
            shape=shape,
            order=order,
        )

    cpdef np.ndarray read_array(
        self,
        dtype,
        int count=-1,
        int offset=0,
        out=None,
    ) except *:
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
        dtype = np.dtype(dtype)

        if count < 0:
            nbytes = self._size if out is None else out.nbytes
            count = nbytes // dtype.itemsize
        else:
            nbytes = count * dtype.itemsize

        result = np.empty(count, dtype) if out is None else out

        if result.nbytes != nbytes:
            raise ValueError('size mismatch')

        assert self._fh is not None

        if offset:
            self._fh.seek(self._offset + offset)

        try:
            n = self._fh.readinto(result)  # type: ignore[attr-defined]
        except AttributeError:
            result[:] = np.frombuffer(self._fh.read(nbytes), dtype).reshape(
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

    cpdef np.recarray read_record(
        self,
        dtype,
        shape=1,
        byteorder=None,
    ) except *:
        """Return NumPy record from file.

        Parameters:
            dtype:
                Data type of record array to read.
            shape:
                Shape of record array to read.
            byteorder:
                Byte order of record array to read.

        """
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
            size = product(sequence(shape)) * dtype.itemsize
            # data = bytearray(size)
            # n = self._fh.readinto(data)
            # data = data[:n]
            # TODO: record is not writable
            data = self._fh.read(size)
            record = np.rec.fromstring(
                data,
                dtype,
                shape,
            )
        return record[0] if shape == 1 else record

    cpdef int write_empty(self, int size) except *:
        """Append null-bytes to file.

        The file position must be at the end of the file.

        Parameters:
            size: Number of null-bytes to write to file.

        """
        if size < 1:
            return 0
        assert self._fh is not None
        self._fh.seek(size - 1, os.SEEK_CUR)
        self._fh.write(b'\x00')
        return size

    cpdef int write_array(
        self,
        np.ndarray data,
        dtype=None,
    ) except *:
        """Write NumPy array to file in C contiguous order.

        Parameters:
            data: Array to write to file.

        """
        assert self._fh is not None
        pos = self._fh.tell()
        # writing non-contiguous arrays is very slow
        data = np.ascontiguousarray(data, dtype)
        try:
            data.tofile(self._fh)
        except io.UnsupportedOperation:
            # numpy cannot write to BytesIO
            self._fh.write(data.tobytes())
        return self._fh.tell() - pos

    cpdef read_segments(
        self,
        Sequence[int] offsets,
        Sequence[int] bytecounts,
        indices=None,
        bint sort=True,
        lock=None,
        buffersize=None,
        bint flat=True,
    ) except *:
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
            lock:
                Reentrant lock to synchronize seeks and reads.
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
        # TODO: Cythonize this?
        assert self._fh is not None
        length = len(offsets)
        if length < 1:
            return
        if length == 1:
            index = 0 if indices is None else indices[0]
            if bytecounts[index] > 0 and offsets[index] > 0:
                if lock is None:
                    lock = self._lock
                with lock:
                    self.seek(offsets[index])
                    data = self._fh.read(bytecounts[index])
            else:
                data = None
            yield (data, index) if flat else [(data, index)]
            return

        if lock is None:
            lock = self._lock
        if buffersize is None:
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

        seek = self.seek
        read = self._fh.read
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
                    with lock:
                        seek(offset)
                        data = read(bytecount)
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
        while i < length:
            result = []
            size = 0
            with lock:
                while size <= buffersize and i < length:
                    index, offset, bytecount = segments[i]
                    if offset > 0 and bytecount > 0:
                        seek(offset)
                        result.append((read(bytecount), index))
                        # buffer = bytearray(bytecount)
                        # n = fh.readinto(buffer)
                        # data.append(buffer[:n])
                        size += bytecount
                    else:
                        result.append((None, index))
                    i += 1
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
        return self._name

    @property
    def dirname(self):
        """Directory in which file is stored."""
        return self._dir

    @property
    def path(self):
        """Absolute path of file."""
        return os.path.join(self._dir, self._name)

    @property
    def extension(self):
        """File name extension of file or stream."""
        name, ext = os.path.splitext(self._name.lower())
        if ext and name.endswith('.ome'):
            ext = '.ome' + ext
        return ext

    @property
    def size(self):
        """Size of file in bytes."""
        return self._size

    @property
    def closed(self):
        """File is closed."""
        return self._fh is None

    @property
    def lock(self):
        """Reentrant lock to synchronize reads and writes."""
        return self._lock

    @lock.setter
    def lock(self, value):
        self.set_lock(value)

    cpdef void set_lock(self, bint value) except *:
        if bool(value) == isinstance(self._lock, NullContext):
            self._lock = threading.RLock() if value else NullContext()

    @property
    def has_lock(self):
        """A reentrant lock is currently used to sync reads and writes."""
        return not isinstance(self._lock, NullContext)

    @property
    def is_file(self):
        """File has fileno and can be memory-mapped."""
        try:
            self._fh.fileno()  # type: ignore[union-attr]
            return True
        except Exception:
            return False


@final
cdef class FileCache:
    """Keep FileHandles open.

    Parameters:
        size: Maximum number of files to keep open. The default is 8.
        lock: Reentrant lock to synchronize reads and writes.

    """

    cdef int size
    """Maximum number of files to keep open."""

    cdef dict files
    """Reference counts of opened files."""

    cdef set keep
    """Set of files to keep open."""

    cdef list past
    """FIFO list of opened files."""

    cdef object lock
    """Reentrant lock to synchronize reads and writes."""

    def __cinit__(self):
        self.past = []
        self.files = {}
        self.keep = set()
        self.size = 8
        self.lock = NullContext()

    def __init__(
        self,
        int size=None,
        lock=None,
    ):
        if size is not None:
            self.size = size
        if lock is not None:
            self.lock = lock

    cpdef void open(self, FileHandle fh) except *:
        """Open file, re-open if necessary."""
        with self.lock:
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

    cpdef void close(self, FileHandle fh) except *:
        """Close least recently used open files."""
        with self.lock:
            if fh in self.files:
                self.files[fh] -= 1
            self._trim()

    cpdef void clear(self) except *:
        """Close all opened files if not in use when opened first."""
        with self.lock:
            for fh, refcount in list(self.files.items()):
                if fh not in self.keep:
                    fh.close()
                    del self.files[fh]
                    del self.past[self.past.index(fh)]

    cpdef bytes read(
        self,
        FileHandle fh,
        int offset,
        int bytecount,
        int whence=0,
    ) except *:
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
        with self.lock:
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

    cpdef int write(
        self,
        FileHandle fh,
        int offset,
        bytes data,
        int whence=0,
    ) except *:
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
        with self.lock:
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

    cpdef void _trim(self) except *:
        """Trim file cache."""
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
        return len(self.files)

    def __repr__(self):
        return f'<tifffile.FileCache @0x{id(self):016X}>'


cdef class FileSequence(Sequence):
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
        **kwargs,
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

    cpdef np.ndarray asarray(
        self,
        imreadargs=None,
        chunkshape=None,
        chunkdtype=None,
        axestiled=None,
        out_inplace=None,
        int ioworkers=1,
        out=None,
        **kwargs,
    ) except *:
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
        # TODO: deprecate kwargs?
        files = self._files
        if imreadargs is not None:
            kwargs |= imreadargs

        if ioworkers is None or ioworkers < 1:
            ioworkers = TIFF.MAXIOWORKERS
        ioworkers = min(len(files), ioworkers)
        assert isinstance(ioworkers, int)  # mypy bug?

        if out_inplace is None and self.imread == imread:
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

    cpdef aszarr(self, **kwargs):
        """Return images from files as Zarr 2 store.

        Parameters:
            **kwargs: Arguments passed to :py:class:`ZarrFileSequenceStore`.

        """
        return ZarrFileSequenceStore(self, **kwargs)

    cpdef void close(self) except *:
        """Close open files."""
        if self._container is not None:
            self._container.close()
        self._container = None

    cpdef str commonpath(self) except *:
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


@final
cdef class TiffSequence(FileSequence):
    r"""Sequence of TIFF files containing compatible array data.

    Same as :py:class:`FileSequence` with the :py:func:`imread` function,
    `'\*.tif'` glob pattern, and `out_inplace` enabled by default.

    """

    def __init__(
        self,
        files=None,
        imread=imread,
        **kwargs,
    ):
        super().__init__(imread, '*.tif' if files is None else files, **kwargs)

    def __repr__(self):
        return f'<tifffile.TiffSequence @0x{id(self):016X}>'
