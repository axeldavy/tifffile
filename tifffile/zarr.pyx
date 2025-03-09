
# TODO: this interface does not expose index keys except in __getitem__
class ZarrStore(MutableMapping[str, bytes]):
    """Zarr 2 store base class.

    ZarrStore instances must be closed with :py:meth:`ZarrStore.close`,
    which is automatically called when using the 'with' context manager.

    Parameters:
        fillvalue:
            Value to use for missing chunks of Zarr 2 store.
            The default is 0.
        chunkmode:
            Specifies how to chunk data.

    References:
        1. https://zarr.readthedocs.io/en/stable/spec/v2.html
        2. https://forum.image.sc/t/multiscale-arrays-v0-1/37930

    """

    _store: dict[str, Any]
    _fillvalue: int | float
    _chunkmode: int

    def __init__(
        self,
        *,
        fillvalue: int | float | None = None,
        chunkmode: CHUNKMODE | int | str | None = None,
    ) -> None:
        self._store = {}
        self._fillvalue = 0 if fillvalue is None else fillvalue
        if chunkmode is None:
            self._chunkmode = CHUNKMODE(0)
        else:
            self._chunkmode = enumarg(CHUNKMODE, chunkmode)

    def __enter__(self) -> ZarrStore:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Close ZarrStore."""

    def flush(self) -> None:
        """Flush ZarrStore."""
        raise PermissionError('ZarrStore is read-only')

    def clear(self) -> None:
        """Clear ZarrStore."""
        raise PermissionError('ZarrStore is read-only')

    def keys(self) -> KeysView[str]:
        """Return keys in ZarrStore."""
        return self._store.keys()

    def items(self) -> ItemsView[str, Any]:
        """Return items in ZarrStore."""
        return self._store.items()

    def values(self) -> ValuesView[Any]:
        """Return values in ZarrStore."""
        return self._store.values()

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object, /) -> bool:
        if key in self._store:
            return True
        assert isinstance(key, str)
        return self._contains(key)

    def _contains(self, key: str, /) -> bool:
        """Return if key is in store."""
        raise NotImplementedError

    def __delitem__(self, key: object, /) -> None:
        raise PermissionError('ZarrStore is read-only')

    def __getitem__(self, key: str, /) -> Any:
        if key in self._store:
            return self._store[key]
        if key[-7:] == '.zarray' or key[-7:] == '.zgroup':
            # catch '.zarray' and 'attribute/.zarray'
            raise KeyError(key)
        return self._getitem(key)

    def _getitem(self, key: str, /) -> NDArray[Any]:
        """Return chunk from file."""
        raise NotImplementedError

    def __setitem__(self, key: str, value: bytes, /) -> None:
        if key in self._store:
            raise KeyError(key)
        if key[-7:] == '.zarray' or key[-7:] == '.zgroup':
            # catch '.zarray' and 'attribute/.zarray'
            raise KeyError(key)
        return self._setitem(key, value)

    def _setitem(self, key: str, value: bytes, /) -> None:
        """Write chunk to file."""
        raise NotImplementedError

    @property
    def is_multiscales(self) -> bool:
        """Return if ZarrStore is multi-scales."""
        return b'multiscales' in self._store['.zattrs']

    @staticmethod
    def _empty_chunk(
        shape: tuple[int, ...],
        dtype: DTypeLike,
        fillvalue: int | float | None,
    ) -> NDArray[Any]:
        """Return empty chunk."""
        if fillvalue is None or fillvalue == 0:
            # return bytes(product(shape) * dtype.itemsize)
            return numpy.zeros(shape, dtype)
        chunk = numpy.empty(shape, dtype)
        chunk[:] = fillvalue
        return chunk  # .tobytes()

    @staticmethod
    def _dtype_str(dtype: numpy.dtype[Any], /) -> str:
        """Return dtype as string with native byte order."""
        if dtype.itemsize == 1:
            byteorder = '|'
        else:
            byteorder = {'big': '>', 'little': '<'}[sys.byteorder]
        return byteorder + dtype.str[1:]

    @staticmethod
    def _json(obj: Any, /) -> bytes:
        """Serialize object to JSON formatted string."""
        return json.dumps(
            obj,
            indent=1,
            sort_keys=True,
            ensure_ascii=True,
            separators=(',', ': '),
        ).encode('ascii')

    @staticmethod
    def _value(value: Any, dtype: numpy.dtype[Any], /) -> Any:
        """Return value which is serializable to JSON."""
        if value is None:
            return value
        if dtype.kind == 'b':
            return bool(value)
        if dtype.kind in 'ui':
            return int(value)
        if dtype.kind == 'f':
            if numpy.isnan(value):
                return 'NaN'
            if numpy.isposinf(value):
                return 'Infinity'
            if numpy.isneginf(value):
                return '-Infinity'
            return float(value)
        if dtype.kind == 'c':
            value = numpy.array(value, dtype)
            return (
                ZarrStore._value(value.real, dtype.type().real.dtype),
                ZarrStore._value(value.imag, dtype.type().imag.dtype),
            )
        return value

    @staticmethod
    def _ndindex(
        shape: tuple[int, ...], chunks: tuple[int, ...], /
    ) -> Iterator[str]:
        """Return iterator over all chunk index strings."""
        assert len(shape) == len(chunks)
        chunked = tuple(
            i // j + (1 if i % j else 0) for i, j in zip(shape, chunks)
        )
        for indices in numpy.ndindex(chunked):
            yield '.'.join(str(index) for index in indices)


@final
class ZarrTiffStore(ZarrStore):
    """Zarr 2 store interface to image array in TiffPage or TiffPageSeries.

    ZarrTiffStore is using a TiffFile instance for reading and decoding chunks.
    Therefore, ZarrTiffStore instances cannot be pickled.

    For writing, image data must be stored in uncompressed, unpredicted,
    and unpacked form. Sparse strips and tiles are not written.

    Parameters:
        arg:
            TIFF page or series to wrap as Zarr 2 store.
        level:
            Pyramidal level to wrap. The default is 0.
        chunkmode:
            Use strips or tiles (0) or whole page data (2) as chunks.
            The default is 0.
        fillvalue:
            Value to use for missing chunks. The default is 0.
        zattrs:
            Additional attributes to store in `.zattrs`.
        multiscales:
            Create a multiscales compatible Zarr 2 group store.
            By default, create a Zarr 2 array store for pages and non-pyramidal
            series.
        lock:
            Reentrant lock to synchronize seeks and reads from file.
            By default, the lock of the parent's file handle is used.
        squeeze:
            Remove length-1 dimensions from shape of TiffPageSeries.
        maxworkers:
            Maximum number of threads to concurrently decode strips or tiles
            if `chunkmode=2`.
            If *None* or *0*, use up to :py:attr:`_TIFF.MAXWORKERS` threads.
        buffersize:
            Approximate number of bytes to read from file in one pass
            if `chunkmode=2`. The default is :py:attr:`_TIFF.BUFFERSIZE`.
        _openfiles:
            Internal API.

    """

    _data: list[TiffPageSeries]
    _filecache: FileCache
    _transform: Callable[[NDArray[Any]], NDArray[Any]] | None
    _maxworkers: int | None
    _buffersize: int | None
    _squeeze: bool | None
    _writable: bool
    _multiscales: bool

    def __init__(
        self,
        arg: TiffPage | TiffFrame | TiffPageSeries,
        *,
        level: int | None = None,
        chunkmode: CHUNKMODE | int | str | None = None,
        fillvalue: int | float | None = None,
        zattrs: dict[str, Any] | None = None,
        multiscales: bool | None = None,
        lock: threading.RLock | NullContext | None = None,
        squeeze: bool | None = None,
        maxworkers: int | None = None,
        buffersize: int | None = None,
        _openfiles: int | None = None,
    ) -> None:
        super().__init__(fillvalue=fillvalue, chunkmode=chunkmode)

        if self._chunkmode not in {0, 2}:
            raise NotImplementedError(f'{self._chunkmode!r} not implemented')

        self._squeeze = None if squeeze is None else bool(squeeze)
        self._maxworkers = maxworkers
        self._buffersize = buffersize

        if isinstance(arg, TiffPageSeries):
            self._data = arg.levels
            self._transform = arg.transform
            if multiscales is not None and not multiscales:
                level = 0
            if level is not None:
                self._data = [self._data[level]]
            name = arg.name
        else:
            self._data = [TiffPageSeries([arg])]
            self._transform = None
            name = 'Unnamed'

        fh = self._data[0].keyframe.parent._parent.filehandle
        self._writable = fh.writable() and self._chunkmode == 0
        if lock is None:
            fh.set_lock(True)
            lock = fh.lock
        self._filecache = FileCache(size=_openfiles, lock=lock)

        zattrs = {} if zattrs is None else dict(zattrs)
        # TODO: Zarr Encoding Specification
        # https://xarray.pydata.org/en/stable/internals/zarr-encoding-spec.html

        if multiscales or len(self._data) > 1:
            # multiscales
            self._multiscales = True
            if '_ARRAY_DIMENSIONS' in zattrs:
                array_dimensions = zattrs.pop('_ARRAY_DIMENSIONS')
            else:
                array_dimensions = list(self._data[0].get_axes(squeeze))
            self._store['.zgroup'] = ZarrStore._json({'zarr_format': 2})
            self._store['.zattrs'] = ZarrStore._json(
                {
                    # TODO: use https://ngff.openmicroscopy.org/latest/
                    'multiscales': [
                        {
                            'version': '0.1',
                            'name': name,
                            'datasets': [
                                {'path': str(i)}
                                for i in range(len(self._data))
                            ],
                            # 'axes': [...]
                            # 'type': 'unknown',
                            'metadata': {},
                        }
                    ],
                    **zattrs,
                }
            )
            shape0 = self._data[0].get_shape(squeeze)
            for level, series in enumerate(self._data):
                keyframe = series.keyframe
                keyframe.decode  # cache decode function
                shape = series.get_shape(squeeze)
                dtype = series.dtype
                if fillvalue is None:
                    self._fillvalue = fillvalue = keyframe.nodata
                if self._chunkmode:
                    chunks = keyframe.shape
                else:
                    chunks = keyframe.chunks
                self._store[f'{level}/.zattrs'] = ZarrStore._json(
                    {
                        '_ARRAY_DIMENSIONS': [
                            (f'{ax}{level}' if i != j else ax)
                            for ax, i, j in zip(
                                array_dimensions, shape, shape0
                            )
                        ]
                    }
                )
                self._store[f'{level}/.zarray'] = ZarrStore._json(
                    {
                        'zarr_format': 2,
                        'shape': shape,
                        'chunks': ZarrTiffStore._chunks(chunks, shape),
                        'dtype': ZarrStore._dtype_str(dtype),
                        'compressor': None,
                        'fill_value': ZarrStore._value(fillvalue, dtype),
                        'order': 'C',
                        'filters': None,
                    }
                )
                if self._writable:
                    self._writable = ZarrTiffStore._is_writable(keyframe)
        else:
            self._multiscales = False
            series = self._data[0]
            keyframe = series.keyframe
            keyframe.decode  # cache decode function
            shape = series.get_shape(squeeze)
            dtype = series.dtype
            if fillvalue is None:
                self._fillvalue = fillvalue = keyframe.nodata
            if self._chunkmode:
                chunks = keyframe.shape
            else:
                chunks = keyframe.chunks
            if '_ARRAY_DIMENSIONS' not in zattrs:
                zattrs['_ARRAY_DIMENSIONS'] = list(series.get_axes(squeeze))
            self._store['.zattrs'] = ZarrStore._json(zattrs)
            self._store['.zarray'] = ZarrStore._json(
                {
                    'zarr_format': 2,
                    'shape': shape,
                    'chunks': ZarrTiffStore._chunks(chunks, shape),
                    'dtype': ZarrStore._dtype_str(dtype),
                    'compressor': None,
                    'fill_value': ZarrStore._value(fillvalue, dtype),
                    'order': 'C',
                    'filters': None,
                }
            )
            if self._writable:
                self._writable = ZarrTiffStore._is_writable(keyframe)

    def close(self) -> None:
        """Close open file handles."""
        if hasattr(self, '_filecache'):
            self._filecache.clear()

    def write_fsspec(
        self,
        jsonfile: str | os.PathLike[Any] | TextIO,
        url: str,
        *,
        groupname: str | None = None,
        templatename: str | None = None,
        compressors: dict[COMPRESSION | int, str | None] | None = None,
        version: int | None = None,
        _shape: Sequence[int] | None = None,
        _axes: Sequence[str] | None = None,
        _index: Sequence[int] | None = None,
        _append: bool = False,
        _close: bool = True,
    ) -> None:
        """Write fsspec ReferenceFileSystem as JSON to file.

        Parameters:
            jsonfile:
                Name or open file handle of output JSON file.
            url:
                Remote location of TIFF file(s) without file name(s).
            groupname:
                Zarr 2 group name.
            templatename:
                Version 1 URL template name. The default is 'u'.
            compressors:
                Mapping of :py:class:`COMPRESSION` codes to Numcodecs codec
                names.
            version:
                Version of fsspec file to write. The default is 0.
            _shape:
                Shape of file sequence (experimental).
            _axes:
                Axes of file sequence (experimental).
            _index
                Index of file in sequence (experimental).
            _append:
                If *True*, only write index keys and values (experimental).
            _close:
                If *True*, no more appends (experimental).

        Raises:
            ValueError:
                ZarrTiffStore cannot be represented as ReferenceFileSystem
                due to features that are not supported by Zarr 2, Numcodecs,
                or Imagecodecs:

                - compressors, such as CCITT
                - filters, such as bitorder reversal, packed integers
                - dtypes, such as float24, complex integers
                - JPEGTables in multi-page series
                - incomplete chunks, such as `imagelength % rowsperstrip != 0`

                Files containing incomplete tiles may fail at runtime.

        Notes:
            Parameters `_shape`,  `_axes`, `_index`, `_append`, and `_close`
            are an experimental API for joining the ReferenceFileSystems of
            multiple files of a TiffSequence.

        References:
            - `fsspec ReferenceFileSystem format
              <https://github.com/fsspec/kerchunk>`_

        """
        compressors = {
            1: None,
            8: 'zlib',
            32946: 'zlib',
            34925: 'lzma',
            50013: 'zlib',  # pixtiff
            5: 'imagecodecs_lzw',
            7: 'imagecodecs_jpeg',
            22610: 'imagecodecs_jpegxr',
            32773: 'imagecodecs_packbits',
            33003: 'imagecodecs_jpeg2k',
            33004: 'imagecodecs_jpeg2k',
            33005: 'imagecodecs_jpeg2k',
            33007: 'imagecodecs_jpeg',
            34712: 'imagecodecs_jpeg2k',
            34887: 'imagecodecs_lerc',
            34892: 'imagecodecs_jpeg',
            34933: 'imagecodecs_png',
            34934: 'imagecodecs_jpegxr',
            48124: 'imagecodecs_jetraw',
            50000: 'imagecodecs_zstd',  # numcodecs.zstd fails w/ unknown sizes
            50001: 'imagecodecs_webp',
            50002: 'imagecodecs_jpegxl',
            52546: 'imagecodecs_jpegxl',
            **({} if compressors is None else compressors),
        }

        for series in self._data:
            errormsg = ' not supported by the fsspec ReferenceFileSystem'
            keyframe = series.keyframe
            if (
                keyframe.compression in {65000, 65001, 65002}
                and keyframe.parent.is_eer
            ):
                compressors[keyframe.compression] = 'imagecodecs_eer'
            if keyframe.compression not in compressors:
                raise ValueError(f'{keyframe.compression!r} is' + errormsg)
            if keyframe.fillorder != 1:
                raise ValueError(f'{keyframe.fillorder!r} is' + errormsg)
            if keyframe.sampleformat not in {1, 2, 3, 6}:
                # TODO: support float24 and cint via filters?
                raise ValueError(f'{keyframe.sampleformat!r} is' + errormsg)
            if (
                keyframe.bitspersample
                not in {
                    8,
                    16,
                    32,
                    64,
                    128,
                }
                and keyframe.compression
                not in {
                    # JPEG
                    7,
                    33007,
                    34892,
                }
                and compressors[keyframe.compression] != 'imagecodecs_eer'
            ):
                raise ValueError(
                    f'BitsPerSample {keyframe.bitspersample} is' + errormsg
                )
            if (
                not self._chunkmode
                and not keyframe.is_tiled
                and keyframe.imagelength % keyframe.rowsperstrip
            ):
                raise ValueError('incomplete chunks are' + errormsg)
            if self._chunkmode and not keyframe.is_final:
                raise ValueError(f'{self._chunkmode!r} is' + errormsg)
            if keyframe.jpegtables is not None and len(series.pages) > 1:
                raise ValueError(
                    'JPEGTables in multi-page files are' + errormsg
                )

        if url is None:
            url = ''
        elif url and url[-1] != '/':
            url += '/'
        url = url.replace('\\', '/')

        if groupname is None:
            groupname = ''
        elif groupname and groupname[-1] != '/':
            groupname += '/'

        byteorder: ByteOrder | None = '<' if sys.byteorder == 'big' else '>'
        if (
            self._data[0].keyframe.parent.byteorder != byteorder
            or self._data[0].keyframe.dtype is None
            or self._data[0].keyframe.dtype.itemsize == 1
        ):
            byteorder = None

        index: str
        _shape = [] if _shape is None else list(_shape)
        _axes = [] if _axes is None else list(_axes)
        if len(_shape) != len(_axes):
            raise ValueError('len(_shape) != len(_axes)')
        if _index is None:
            index = ''
        elif len(_shape) != len(_index):
            raise ValueError('len(_shape) != len(_index)')
        elif _index:
            index = '.'.join(str(i) for i in _index)
            index += '.'

        refs: dict[str, Any] = {}
        refzarr: dict[str, Any]
        if version == 1:
            if _append:
                raise ValueError('cannot append to version 1')
            if templatename is None:
                templatename = 'u'
            refs['version'] = 1
            refs['templates'] = {}
            refs['gen'] = []
            templates = {}
            if self._data[0].is_multifile:
                i = 0
                for page in self._data[0].pages:
                    if page is None or page.keyframe is None:
                        continue
                    fname = page.keyframe.parent.filehandle.name
                    if fname in templates:
                        continue
                    key = f'{templatename}{i}'
                    templates[fname] = f'{{{{{key}}}}}'
                    refs['templates'][key] = url + fname
                    i += 1
            else:
                fname = self._data[0].keyframe.parent.filehandle.name
                key = f'{templatename}'
                templates[fname] = f'{{{{{key}}}}}'
                refs['templates'][key] = url + fname

            refs['refs'] = refzarr = {}
        else:
            refzarr = refs

        if not _append:
            if groupname:
                # TODO: support nested groups
                refzarr['.zgroup'] = ZarrStore._json(
                    {'zarr_format': 2}
                ).decode()

            for key, value in self._store.items():
                if '.zattrs' in key and _axes:
                    value = json.loads(value)
                    if '_ARRAY_DIMENSIONS' in value:
                        value['_ARRAY_DIMENSIONS'] = (
                            _axes + value['_ARRAY_DIMENSIONS']
                        )
                    value = ZarrStore._json(value)
                elif '.zarray' in key:
                    level = int(key.split('/')[0]) if '/' in key else 0
                    keyframe = self._data[level].keyframe
                    value = json.loads(value)
                    if _shape:
                        value['shape'] = _shape + value['shape']
                        value['chunks'] = [1] * len(_shape) + value['chunks']
                    codec_id = compressors[keyframe.compression]
                    if codec_id == 'imagecodecs_jpeg':
                        # TODO: handle JPEG color spaces
                        jpegtables = keyframe.jpegtables
                        if jpegtables is None:
                            tables = None
                        else:
                            import base64

                            tables = base64.b64encode(jpegtables).decode()
                        jpegheader = keyframe.jpegheader
                        if jpegheader is None:
                            header = None
                        else:
                            import base64

                            header = base64.b64encode(jpegheader).decode()
                        (
                            colorspace_jpeg,
                            colorspace_data,
                        ) = jpeg_decode_colorspace(
                            keyframe.photometric,
                            keyframe.planarconfig,
                            keyframe.extrasamples,
                            keyframe.is_jfif,
                        )
                        value['compressor'] = {
                            'id': codec_id,
                            'tables': tables,
                            'header': header,
                            'bitspersample': keyframe.bitspersample,
                            'colorspace_jpeg': colorspace_jpeg,
                            'colorspace_data': colorspace_data,
                        }
                    elif (
                        codec_id == 'imagecodecs_webp'
                        and keyframe.samplesperpixel == 4
                    ):
                        value['compressor'] = {
                            'id': codec_id,
                            'hasalpha': True,
                        }
                    elif codec_id == 'imagecodecs_eer':
                        if keyframe.compression == 65002:
                            rlebits = int(keyframe.tags.valueof(65007, 7))
                            horzbits = int(keyframe.tags.valueof(65008, 2))
                            vertbits = int(keyframe.tags.valueof(65009, 2))
                        elif keyframe.compression == 65001:
                            rlebits = 7
                            horzbits = 2
                            vertbits = 2
                        else:
                            rlebits = 8
                            horzbits = 2
                            vertbits = 2
                        value['compressor'] = {
                            'id': codec_id,
                            'shape': keyframe.chunks,
                            'rlebits': rlebits,
                            'horzbits': horzbits,
                            'vertbits': vertbits,
                        }
                    elif codec_id is not None:
                        value['compressor'] = {'id': codec_id}
                    if byteorder is not None:
                        value['dtype'] = byteorder + value['dtype'][1:]
                    if keyframe.predictor > 1:
                        # predictors need access to chunk shape and dtype
                        # requires imagecodecs > 2021.8.26 to read
                        if keyframe.predictor in {2, 34892, 34893}:
                            filter_id = 'imagecodecs_delta'
                        else:
                            filter_id = 'imagecodecs_floatpred'
                        if keyframe.predictor <= 3:
                            dist = 1
                        elif keyframe.predictor in {34892, 34894}:
                            dist = 2
                        else:
                            dist = 4
                        if (
                            keyframe.planarconfig == 1
                            and keyframe.samplesperpixel > 1
                        ):
                            axis = -2
                        else:
                            axis = -1
                        value['filters'] = [
                            {
                                'id': filter_id,
                                'axis': axis,
                                'dist': dist,
                                'shape': value['chunks'],
                                'dtype': value['dtype'],
                            }
                        ]
                    value = ZarrStore._json(value)

                refzarr[groupname + key] = value.decode()

        fh: TextIO
        if hasattr(jsonfile, 'write'):
            fh = jsonfile  # type: ignore[assignment]
        else:
            fh = open(jsonfile, 'w', encoding='utf-8')

        if version == 1:
            fh.write(json.dumps(refs, indent=1).rsplit('}"', 1)[0] + '}"')
            indent = '  '
        elif _append:
            indent = ' '
        else:
            fh.write(json.dumps(refs, indent=1)[:-2])
            indent = ' '

        for key, value in self._store.items():
            if '.zarray' in key:
                value = json.loads(value)
                shape = value['shape']
                chunks = value['chunks']
                levelstr = (key.split('/')[0] + '/') if '/' in key else ''
                for chunkindex in ZarrStore._ndindex(shape, chunks):
                    key = levelstr + chunkindex
                    keyframe, page, _, offset, bytecount = self._parse_key(key)
                    key = levelstr + index + chunkindex
                    if page and self._chunkmode and offset is None:
                        offset = page.dataoffsets[0]
                        bytecount = keyframe.nbytes
                    if offset and bytecount:
                        fname = keyframe.parent.filehandle.name
                        if version == 1:
                            fname = templates[fname]
                        else:
                            fname = f'{url}{fname}'
                        fh.write(
                            f',\n{indent}"{groupname}{key}": '
                            f'["{fname}", {offset}, {bytecount}]'
                        )

        # TODO: support nested groups
        if version == 1:
            fh.write('\n }\n}')
        elif _close:
            fh.write('\n}')

        if not hasattr(jsonfile, 'write'):
            fh.close()

    def _contains(self, key: str, /) -> bool:
        """Return if key is in store."""
        try:
            _, page, _, offset, bytecount = self._parse_key(key)
        except (KeyError, IndexError):
            return False
        if self._chunkmode and offset is None:
            return True
        return (
            page is not None
            and offset is not None
            and bytecount is not None
            and offset > 0
            and bytecount > 0
        )

    def _getitem(self, key: str, /) -> NDArray[Any]:
        """Return chunk from file."""
        keyframe, page, chunkindex, offset, bytecount = self._parse_key(key)

        if page is None or offset == 0 or bytecount == 0:
            raise KeyError(key)

        fh = page.parent.filehandle

        if self._chunkmode:
            if offset is not None:
                # contiguous image data in page or series
                # create virtual frame instead of loading page from file
                assert bytecount is not None
                page = TiffFrame(
                    page.parent,
                    index=0,
                    keyframe=keyframe,
                    dataoffsets=(offset,),
                    databytecounts=(bytecount,),
                )
            self._filecache.open(fh)
            chunk = page.asarray(
                lock=self._filecache.lock,
                maxworkers=self._maxworkers,
                buffersize=self._buffersize,
            )
            self._filecache.close(fh)
            if self._transform is not None:
                chunk = self._transform(chunk)
            return chunk

        assert offset is not None and bytecount is not None
        chunk_bytes = self._filecache.read(fh, offset, bytecount)

        decodeargs: dict[str, Any] = {'_fullsize': True}
        if page.jpegtables is not None:
            decodeargs['jpegtables'] = page.jpegtables
        if keyframe.jpegheader is not None:
            decodeargs['jpegheader'] = keyframe.jpegheader

        assert chunkindex is not None
        chunk = keyframe.decode(
            chunk_bytes, chunkindex, **decodeargs  # type: ignore[assignment]
        )[0]
        assert chunk is not None
        if self._transform is not None:
            chunk = self._transform(chunk)

        if self._chunkmode:
            chunks = keyframe.shape
        else:
            chunks = keyframe.chunks
        if chunk.size != product(chunks):
            raise RuntimeError(f'{chunk.size} != {product(chunks)}')
        return chunk  # .tobytes()

    def _setitem(self, key: str, value: bytes, /) -> None:
        """Write chunk to file."""
        if not self._writable:
            raise PermissionError('ZarrStore is read-only')
        keyframe, page, chunkindex, offset, bytecount = self._parse_key(key)
        if (
            page is None
            or offset is None
            or offset == 0
            or bytecount is None
            or bytecount == 0
        ):
            return
        if bytecount < len(value):
            value = value[:bytecount]
        self._filecache.write(page.parent.filehandle, offset, value)

    def _parse_key(self, key: str, /) -> tuple[
        TiffPage,
        TiffPage | TiffFrame | None,
        int | None,
        int | None,
        int | None,
    ]:
        """Return keyframe, page, index, offset, and bytecount from key.

        Raise KeyError if key is not valid.

        """
        if self._multiscales:
            try:
                level, key = key.split('/')
                series = self._data[int(level)]
            except (ValueError, IndexError) as exc:
                raise KeyError(key) from exc
        else:
            series = self._data[0]
        keyframe = series.keyframe
        pageindex, chunkindex = self._indices(key, series)
        if series.dataoffset is not None:
            # contiguous or truncated
            page = series[0]
            if page is None or page.dtype is None or page.keyframe is None:
                return keyframe, None, chunkindex, 0, 0
            offset = pageindex * page.size * page.dtype.itemsize
            try:
                offset += page.dataoffsets[chunkindex]
            except IndexError as exc:
                raise KeyError(key) from exc
            if self._chunkmode:
                bytecount = page.size * page.dtype.itemsize
                return page.keyframe, page, chunkindex, offset, bytecount
        elif self._chunkmode:
            with self._filecache.lock:
                page = series[pageindex]
            if page is None or page.keyframe is None:
                return keyframe, None, None, 0, 0
            return page.keyframe, page, None, None, None
        else:
            with self._filecache.lock:
                page = series[pageindex]
            if page is None or page.keyframe is None:
                return keyframe, None, chunkindex, 0, 0
            try:
                offset = page.dataoffsets[chunkindex]
            except IndexError:
                # raise KeyError(key) from exc
                # issue #249: Philips may be missing last row of tiles
                return page.keyframe, page, chunkindex, 0, 0
        try:
            bytecount = page.databytecounts[chunkindex]
        except IndexError as exc:
            raise KeyError(key) from exc
        return page.keyframe, page, chunkindex, offset, bytecount

    def _indices(self, key: str, series: TiffPageSeries, /) -> tuple[int, int]:
        """Return page and strile indices from Zarr chunk index."""
        keyframe = series.keyframe
        shape = series.get_shape(self._squeeze)
        try:
            indices = [int(i) for i in key.split('.')]
        except ValueError as exc:
            raise KeyError(key) from exc
        assert len(indices) == len(shape)
        if self._chunkmode:
            chunked = (1,) * len(keyframe.shape)
        else:
            chunked = keyframe.chunked
        p = 1
        for i, s in enumerate(shape[::-1]):
            p *= s
            if p == keyframe.size:
                i = len(indices) - i - 1
                frames_indices = indices[:i]
                strile_indices = indices[i:]
                frames_chunked = shape[:i]
                strile_chunked = list(shape[i:])  # updated later
                break
        else:
            raise RuntimeError
        if len(strile_chunked) == len(keyframe.shape):
            strile_chunked = list(chunked)
        else:
            # get strile_chunked including singleton dimensions
            i = len(strile_indices) - 1
            j = len(keyframe.shape) - 1
            while True:
                if strile_chunked[i] == keyframe.shape[j]:
                    strile_chunked[i] = chunked[j]
                    i -= 1
                    j -= 1
                elif strile_chunked[i] == 1:
                    i -= 1
                else:
                    raise RuntimeError('shape does not match page shape')
                if i < 0 or j < 0:
                    break
            assert product(strile_chunked) == product(chunked)
        if len(frames_indices) > 0:
            frameindex = int(
                numpy.ravel_multi_index(frames_indices, frames_chunked)
            )
        else:
            frameindex = 0
        if len(strile_indices) > 0:
            strileindex = int(
                numpy.ravel_multi_index(strile_indices, strile_chunked)
            )
        else:
            strileindex = 0
        return frameindex, strileindex

    @staticmethod
    def _chunks(
        chunks: tuple[int, ...], shape: tuple[int, ...], /
    ) -> tuple[int, ...]:
        """Return chunks with same length as shape."""
        ndim = len(shape)
        if ndim == 0:
            return ()  # empty array
        if 0 in shape:
            return (1,) * ndim
        newchunks = []
        i = ndim - 1
        j = len(chunks) - 1
        while True:
            if j < 0:
                newchunks.append(1)
                i -= 1
            elif shape[i] > 1 and chunks[j] > 1:
                newchunks.append(chunks[j])
                i -= 1
                j -= 1
            elif shape[i] == chunks[j]:  # both 1
                newchunks.append(1)
                i -= 1
                j -= 1
            elif shape[i] == 1:
                newchunks.append(1)
                i -= 1
            elif chunks[j] == 1:
                newchunks.append(1)
                j -= 1
            else:
                raise RuntimeError
            if i < 0 or ndim == len(newchunks):
                break
        # assert ndim == len(newchunks)
        return tuple(newchunks[::-1])

    @staticmethod
    def _is_writable(keyframe: TiffPage) -> bool:
        """Return True if chunks are writable."""
        return (
            keyframe.compression == 1
            and keyframe.fillorder == 1
            and keyframe.sampleformat in {1, 2, 3, 6}
            and keyframe.bitspersample in {8, 16, 32, 64, 128}
            # and (
            #     keyframe.rowsperstrip == 0
            #     or keyframe.imagelength % keyframe.rowsperstrip == 0
            # )
        )

    def __enter__(self) -> ZarrTiffStore:
        return self

    def __repr__(self) -> str:
        return f'<tifffile.ZarrTiffStore @0x{id(self):016X}>'


@final
class ZarrFileSequenceStore(ZarrStore):
    """Zarr 2 store interface to image array in FileSequence.

    Parameters:
        filesequence:
            FileSequence instance to wrap as Zarr 2 store.
            Files in containers are not supported.
        fillvalue:
            Value to use for missing chunks. The default is 0.
        chunkmode:
            Currently only one chunk per file is supported.
        chunkshape:
            Shape of chunk in each file.
            Must match ``FileSequence.imread(file, **imreadargs).shape``.
        chunkdtype:
            Data type of chunk in each file.
            Must match ``FileSequence.imread(file, **imreadargs).dtype``.
        axestiled:
            Axes to be tiled. Map stacked sequence axis to chunk axis.
        zattrs:
            Additional attributes to store in `.zattrs`.
        imreadargs:
            Arguments passed to :py:attr:`FileSequence.imread`.
        **kwargs:
            Arguments passed to :py:attr:`FileSequence.imread`in addition
            to `imreadargs`.

    Notes:
        If `chunkshape` or `chunkdtype` are *None* (default), their values
        are determined by reading the first file with
        ``FileSequence.imread(arg.files[0], **imreadargs)``.

    """

    imread: Callable[..., NDArray[Any]]
    """Function to read image array from single file."""

    _lookup: dict[tuple[int, ...], str]
    _chunks: tuple[int, ...]
    _dtype: numpy.dtype[Any]
    _tiled: TiledSequence
    _commonpath: str
    _kwargs: dict[str, Any]

    def __init__(
        self,
        filesequence: FileSequence,
        *,
        fillvalue: int | float | None = None,
        chunkmode: CHUNKMODE | int | str | None = None,
        chunkshape: Sequence[int] | None = None,
        chunkdtype: DTypeLike | None = None,
        axestiled: dict[int, int] | Sequence[tuple[int, int]] | None = None,
        zattrs: dict[str, Any] | None = None,
        imreadargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(fillvalue=fillvalue, chunkmode=chunkmode)

        if self._chunkmode not in {0, 3}:
            raise ValueError(f'invalid chunkmode {self._chunkmode!r}')

        if not isinstance(filesequence, FileSequence):
            raise TypeError('not a FileSequence')

        if filesequence._container:
            raise NotImplementedError('cannot open container as Zarr 2 store')

        # TODO: deprecate kwargs?
        if imreadargs is not None:
            kwargs |= imreadargs

        self._kwargs = kwargs
        self._imread = filesequence.imread
        self._commonpath = filesequence.commonpath()

        if chunkshape is None or chunkdtype is None:
            chunk = filesequence.imread(filesequence[0], **kwargs)
            self._chunks = chunk.shape
            self._dtype = chunk.dtype
        else:
            self._chunks = tuple(chunkshape)
            self._dtype = numpy.dtype(chunkdtype)
            chunk = None

        self._tiled = TiledSequence(
            filesequence.shape, self._chunks, axestiled=axestiled
        )
        self._lookup = dict(
            zip(self._tiled.indices(filesequence.indices), filesequence)
        )

        zattrs = {} if zattrs is None else dict(zattrs)
        # TODO: add _ARRAY_DIMENSIONS to ZarrFileSequenceStore
        # if '_ARRAY_DIMENSIONS' not in zattrs:
        #     zattrs['_ARRAY_DIMENSIONS'] = list(...)

        self._store['.zattrs'] = ZarrStore._json(zattrs)
        self._store['.zarray'] = ZarrStore._json(
            {
                'zarr_format': 2,
                'shape': self._tiled.shape,
                'chunks': self._tiled.chunks,
                'dtype': ZarrStore._dtype_str(self._dtype),
                'compressor': None,
                'fill_value': ZarrStore._value(fillvalue, self._dtype),
                'order': 'C',
                'filters': None,
            }
        )

    def _contains(self, key: str, /) -> bool:
        """Return if key is in store."""
        try:
            indices = tuple(int(i) for i in key.split('.'))
        except Exception:
            return False
        return indices in self._lookup

    def _getitem(self, key: str, /) -> NDArray[Any]:
        """Return chunk from file."""
        indices = tuple(int(i) for i in key.split('.'))
        filename = self._lookup.get(indices, None)
        if filename is None:
            raise KeyError(key)
        return self._imread(filename, **self._kwargs)

    def _setitem(self, key: str, value: bytes, /) -> None:
        raise PermissionError('ZarrStore is read-only')

    def write_fsspec(
        self,
        jsonfile: str | os.PathLike[Any] | TextIO,
        url: str,
        *,
        quote: bool | None = None,
        groupname: str | None = None,
        templatename: str | None = None,
        codec_id: str | None = None,
        version: int | None = None,
        _append: bool = False,
        _close: bool = True,
    ) -> None:
        """Write fsspec ReferenceFileSystem as JSON to file.

        Parameters:
            jsonfile:
                Name or open file handle of output JSON file.
            url:
                Remote location of TIFF file(s) without file name(s).
            quote:
                Quote file names, that is, replace ' ' with '%20'.
                The default is True.
            groupname:
                Zarr 2 group name.
            templatename:
                Version 1 URL template name. The default is 'u'.
            codec_id:
                Name of Numcodecs codec to decode files or chunks.
            version:
                Version of fsspec file to write. The default is 0.
            _append, _close:
                Experimental API.

        References:
            - `fsspec ReferenceFileSystem format
              <https://github.com/fsspec/kerchunk>`_

        """
        from urllib.parse import quote as quote_

        kwargs = self._kwargs.copy()

        if codec_id is not None:
            pass
        elif self._imread == imread:
            codec_id = 'tifffile'
        elif 'imagecodecs.' in self._imread.__module__:
            if (
                self._imread.__name__ != 'imread'
                or 'codec' not in self._kwargs
            ):
                raise ValueError('cannot determine codec_id')
            codec = kwargs.pop('codec')
            if isinstance(codec, (list, tuple)):
                codec = codec[0]
            if callable(codec):
                codec = codec.__name__.split('_')[0]
            codec_id = {
                'apng': 'imagecodecs_apng',
                'avif': 'imagecodecs_avif',
                'gif': 'imagecodecs_gif',
                'heif': 'imagecodecs_heif',
                'jpeg': 'imagecodecs_jpeg',
                'jpeg8': 'imagecodecs_jpeg',
                'jpeg12': 'imagecodecs_jpeg',
                'jpeg2k': 'imagecodecs_jpeg2k',
                'jpegls': 'imagecodecs_jpegls',
                'jpegxl': 'imagecodecs_jpegxl',
                'jpegxr': 'imagecodecs_jpegxr',
                'ljpeg': 'imagecodecs_ljpeg',
                'lerc': 'imagecodecs_lerc',
                # 'npy': 'imagecodecs_npy',
                'png': 'imagecodecs_png',
                'qoi': 'imagecodecs_qoi',
                'tiff': 'imagecodecs_tiff',
                'webp': 'imagecodecs_webp',
                'zfp': 'imagecodecs_zfp',
            }[codec]
        else:
            # TODO: choose codec from filename
            raise ValueError('cannot determine codec_id')

        if url is None:
            url = ''
        elif url and url[-1] != '/':
            url += '/'

        if groupname is None:
            groupname = ''
        elif groupname and groupname[-1] != '/':
            groupname += '/'

        refs: dict[str, Any] = {}
        if version == 1:
            if _append:
                raise ValueError('cannot append to version 1 files')
            if templatename is None:
                templatename = 'u'
            refs['version'] = 1
            refs['templates'] = {templatename: url}
            refs['gen'] = []
            refs['refs'] = refzarr = {}
            url = f'{{{{{templatename}}}}}'
        else:
            refzarr = refs

        if groupname and not _append:
            refzarr['.zgroup'] = ZarrStore._json({'zarr_format': 2}).decode()

        for key, value in self._store.items():
            if '.zarray' in key:
                value = json.loads(value)
                # TODO: make kwargs serializable
                value['compressor'] = {'id': codec_id, **kwargs}
                value = ZarrStore._json(value)
            refzarr[groupname + key] = value.decode()

        fh: TextIO
        if hasattr(jsonfile, 'write'):
            fh = jsonfile  # type: ignore[assignment]
        else:
            fh = open(jsonfile, 'w', encoding='utf-8')

        if version == 1:
            fh.write(json.dumps(refs, indent=1).rsplit('}"', 1)[0] + '}"')
            indent = '  '
        elif _append:
            fh.write(',\n')
            fh.write(json.dumps(refs, indent=1)[2:-2])
            indent = ' '
        else:
            fh.write(json.dumps(refs, indent=1)[:-2])
            indent = ' '

        prefix = len(self._commonpath)

        for key, value in self._store.items():
            if '.zarray' in key:
                value = json.loads(value)
                for index, filename in sorted(
                    self._lookup.items(), key=lambda x: x[0]
                ):
                    filename = filename[prefix:].replace('\\', '/')
                    if quote is None or quote:
                        filename = quote_(filename)
                    if filename[0] == '/':
                        filename = filename[1:]
                    indexstr = '.'.join(str(i) for i in index)
                    fh.write(
                        f',\n{indent}"{groupname}{indexstr}": '
                        f'["{url}{filename}"]'
                    )

        if version == 1:
            fh.write('\n }\n}')
        elif _close:
            fh.write('\n}')

        if not hasattr(jsonfile, 'write'):
            fh.close()

    def __enter__(self) -> ZarrFileSequenceStore:
        return self

    def __repr__(self) -> str:
        return f'<tifffile.ZarrFileSequenceStore @0x{id(self):016X}>'

    def __str__(self) -> str:
        return '\n '.join(
            (
                self.__class__.__name__,
                'shape: {}'.format(
                    ', '.join(str(i) for i in self._tiled.shape)
                ),
                'chunks: {}'.format(
                    ', '.join(str(i) for i in self._tiled.chunks)
                ),
                f'dtype: {self._dtype}',
                f'fillvalue: {self._fillvalue}',
            )
        )

def zarr_selection(
    store: ZarrStore,
    selection: Any,
    *,
    groupindex: int | None = None,
    close: bool = True,
    out: OutputType = None,
) -> NDArray[Any]:
    """Return selection from Zarr 2 store.

    Parameters:
        store:
            ZarrStore instance to read selection from.
        selection:
            Subset of image to be extracted and returned.
            Refer to the Zarr 2 documentation for valid selections.
        groupindex:
            Index of array if store is Zarr 2 group.
        close:
            Close store before returning.
        out:
            Specifies how image array is returned.
            By default, create a new array.
            If a *numpy.ndarray*, a writable array to which the images
            are copied.
            If *'memmap'*, create a memory-mapped array in a temporary
            file.
            If a *string* or *open file*, the file used to create a
            memory-mapped array.

    """
    import zarr

    z = zarr.open(store, mode='r')
    try:
        if isinstance(z, zarr.hierarchy.Group):
            if groupindex is None:
                groupindex = 0
            z = z[groupindex]
        if out is not None:
            shape = zarr.indexing.BasicIndexer(selection, z).shape
            out = create_output(out, shape, z.dtype)
        result = z.get_basic_selection(selection, out=out)
    finally:
        if close:
            store.close()
    return result