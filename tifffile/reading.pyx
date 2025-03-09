
cdef class TiffFile:
    """Read image and metadata from TIFF file.

    TiffFile instances must be closed with :py:meth:`TiffFile.close`, which
    is automatically called when using the 'with' context manager.

    TiffFile instances are not thread-safe. All attributes are read-only.

    Parameters:
        file:
            Specifies TIFF file to read.
            Open file objects must be positioned at the TIFF header.
        mode:
            File open mode if `file` is file name. The default is 'rb'.
        name:
            Name of file if `file` is file handle.
        offset:
            Start position of embedded file.
            The default is the current file position.
        size:
            Size of embedded file. The default is the number of bytes
            from the `offset` to the end of the file.
        omexml:
            OME metadata in XML format, for example, from external companion
            file or sanitized XML overriding XML in file.
        _multifile, _useframes, _parent:
            Internal use.
        **is_flags:
            Override `TiffFile.is_` flags, for example:

            ``is_ome=False``: disable processing of OME-XML metadata.
            ``is_lsm=False``: disable special handling of LSM files.
            ``is_ndpi=True``: force file to be NDPI format.

    Raises:
        TiffFileError: Invalid TIFF structure.

    """

    cdef TiffFormat tiff_format
    """Properties of TIFF file format."""

    cdef TiffPages pages
    """Sequence of pages in TIFF file."""

    cdef FileHandle _fh
    cdef bint _multifile
    cdef TiffFile _parent  # OME master file
    cdef dict _files # cache of TiffFile instances, dict[str | None, TiffFile] 
    cdef str _omexml  # external OME-XML
    cdef dict _decoders # cache of TiffPage.decode functions

    cdef bint is_ome

    def __init__(
        self,
        object file,
        *,
        str mode = None,
        str name = None,
        int64_t offset = -1,
        int64_t size = -1,
        str omexml = None,
        bint _multifile = True,
        bint _useframes = False,
        TiffFile _parent = None,
        **is_flags,
    ) -> None:
        #for key, value in is_flags.items():
        #    if key[:3] == 'is_' and key[3:] in TIFF.FILE_FLAGS:
        #        if value is not None:
        #            setattr(self, key, bool(value))
        #    else:
        #        raise TypeError(f'unexpected keyword argument: {key}')

        if mode not in {None, 'r', 'r+', 'rb', 'r+b'}:
            raise ValueError(f'invalid mode {mode!r}')

        self._omexml = None
        if omexml:
            if omexml.strip()[-4:] != 'OME>':
                raise ValueError('invalid OME-XML')
            self._omexml = omexml
            self.is_ome = True

        fh = FileHandle(file, mode=mode, name=name, offset=offset, size=size)
        self._fh = fh
        self._multifile = _multifile
        self._files = {fh.name: self}
        self._decoders = {}
        self._parent = self if _parent is None else _parent

        fh.seek(0)
        # Read maximum header size
        header = fh.read(32)
        if len(header) < 8:
            raise TiffFileError(f'not a TIFF file {header!r}')
        self.tiff_format = TiffFormat.detect_format(header)
        self.pages = TiffPages(self)

            if self.is_lsm and (
                self.filehandle.size >= 2**32
                or self.pages[0].compression != 1
                or self.pages[1].compression != 1
            ):
                self._lsm_load_pages()

            elif self.is_scanimage and not self.is_bigtiff:
                # ScanImage <= 2015
                try:
                    self.pages._load_virtual_frames()
                except Exception as exc:
                    logger().error(
                        f'{self!r} <TiffPages._load_virtual_frames> '
                        f'raised {exc!r:.128}'
                    )

            elif self.is_ndpi:
                try:
                    self._ndpi_load_pages()
                except Exception as exc:
                    logger().error(
                        f'{self!r} <_ndpi_load_pages> raised {exc!r:.128}'
                    )

            elif _useframes:
                self.pages.useframes = True

    def __del__(self) -> None:
        self.close()

    @property
    def byteorder(self) -> Literal['>', '<']:
        """Byteorder of TIFF file."""
        return '>' if self.tiff_format.byteorder == BYTEORDER.II else '<' # TODO: check if this is correct

    @property
    def filehandle(self) -> FileHandle:
        """File handle."""
        return self._fh

    @property
    def filename(self) -> str:
        """Name of file handle."""
        return self._fh.name

    @cached_property
    def fstat(self) -> Any:
        """Status of file handle's descriptor, if any."""
        try:
            return os.fstat(self._fh.fileno())
        except Exception:  # io.UnsupportedOperation
            return None

    def close(self) -> None:
        """Close open file handle(s)."""
        for tif in self._files.values():
            tif.filehandle.close()

    def asarray(
        self,
        key: int | slice | Iterable[int] | None = None,
        *,
        series: int | TiffPageSeries | None = None,
        level: int | None = None,
        squeeze: bool | None = None,
        out: OutputType = None,
        maxworkers: int | None = None,
        buffersize: int | None = None,
    ) -> NDArray[Any]:
        """Return images from select pages as NumPy array.

        By default, the image array from the first level of the first series
        is returned.

        Parameters:
            key:
                Specifies which pages to return as array.
                By default, the image of the specified `series` and `level`
                is returned.
                If not *None*, the images from the specified pages in the
                whole file (if `series` is *None*) or a specified series are
                returned as a stacked array.
                Requesting an array from multiple pages that are not
                compatible wrt. shape, dtype, compression etc. is undefined,
                that is, it may crash or return incorrect values.
            series:
                Specifies which series of pages to return as array.
                The default is 0.
            level:
                Specifies which level of multi-resolution series to return
                as array. The default is 0.
            squeeze:
                If *True*, remove all length-1 dimensions (except X and Y)
                from array.
                If *False*, single pages are returned as 5D array of shape
                :py:attr:`TiffPage.shaped`.
                For series, the shape of the returned array also includes
                singlet dimensions specified in some file formats.
                For example, ImageJ series and most commonly also OME series,
                are returned in TZCYXS order.
                By default, all but `"shaped"` series are squeezed.
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
            maxworkers:
                Maximum number of threads to concurrently decode data from
                multiple pages or compressed segments.
                If *None* or *0*, use up to :py:attr:`_TIFF.MAXWORKERS`
                threads. Reading data from file is limited to the main thread.
                Using multiple threads can significantly speed up this
                function if the bottleneck is decoding compressed data,
                for example, in case of large LZW compressed LSM files or
                JPEG compressed tiled slides.
                If the bottleneck is I/O or pure Python code, using multiple
                threads might be detrimental.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.

        Returns:
            Images from specified pages. See `TiffPage.asarray`
            for operations that are applied (or not) to the image data
            stored in the file.

        """
        if not self.pages:
            return numpy.array([])
        if key is None and series is None:
            series = 0

        pages: Any  # TiffPages | TiffPageSeries | list[TiffPage | TiffFrame]
        page0: TiffPage | TiffFrame | None

        if series is None:
            pages = self.pages
        else:
            if not isinstance(series, TiffPageSeries):
                series = self.series[series]
            if level is not None:
                series = series.levels[level]
            pages = series

        if key is None:
            pass
        elif series is None:
            pages = pages._getlist(key)
        elif isinstance(key, (int, numpy.integer)):
            pages = [pages[int(key)]]
        elif isinstance(key, slice):
            pages = pages[key]
        elif isinstance(key, Iterable) and not isinstance(key, str):
            pages = [pages[k] for k in key]
        else:
            raise TypeError(
                f'key must be an integer, slice, or sequence, not {type(key)}'
            )

        if pages is None or len(pages) == 0:
            raise ValueError('no pages selected')

        if (
            key is None
            and series is not None
            and series.dataoffset is not None
        ):
            typecode = self.byteorder + series.dtype.char
            if (
                series.keyframe.is_memmappable
                and isinstance(out, str)
                and out == 'memmap'
            ):
                # direct mapping
                shape = series.get_shape(squeeze)
                result = self.filehandle.memmap_array(
                    typecode, shape, series.dataoffset
                )
            else:
                # read into output
                shape = series.get_shape(squeeze)
                if out is not None:
                    out = create_output(out, shape, series.dtype)
                result = self.filehandle.read_array(
                    typecode,
                    series.size,
                    series.dataoffset,
                    out=out,
                )
        elif len(pages) == 1:
            page0 = pages[0]
            if page0 is None:
                raise ValueError('page is None')
            result = page0.asarray(
                out=out, maxworkers=maxworkers, buffersize=buffersize
            )
        else:
            result = stack_pages(
                pages, out=out, maxworkers=maxworkers, buffersize=buffersize
            )

        assert result is not None

        if key is None:
            assert series is not None  # TODO: ?
            shape = series.get_shape(squeeze)
            try:
                result.shape = shape
            except ValueError as exc:
                try:
                    logger().warning(
                        f'{self!r} <asarray> failed to reshape '
                        f'{result.shape} to {shape}, raised {exc!r:.128}'
                    )
                    # try series of expected shapes
                    result.shape = (-1,) + shape
                except ValueError:
                    # revert to generic shape
                    result.shape = (-1,) + series.keyframe.shape
        elif len(pages) == 1:
            if squeeze is None:
                squeeze = True
            page0 = pages[0]
            if page0 is None:
                raise ValueError('page is None')
            result.shape = page0.shape if squeeze else page0.shaped
        else:
            if squeeze is None:
                squeeze = True
            try:
                page0 = next(p for p in pages if p is not None)
            except StopIteration as exc:
                raise ValueError('pages are all None') from exc
            assert page0 is not None
            result.shape = (-1,) + (page0.shape if squeeze else page0.shaped)
        return result

    def aszarr(
        self,
        key: int | None = None,
        *,
        series: int | TiffPageSeries | None = None,
        level: int | None = None,
        **kwargs: Any,
    ) -> ZarrTiffStore:
        """Return images from select pages as Zarr 2 store.

        By default, the images from the first series, including all levels,
        are wrapped as a Zarr 2 store.

        Parameters:
            key:
                Index of page in file (if `series` is None) or series to wrap
                as Zarr 2 store.
                By default, a series is wrapped.
            series:
                Index of series to wrap as Zarr 2 store.
                The default is 0 (if `key` is None).
            level:
                Index of pyramid level in series to wrap as Zarr 2 store.
                By default, all levels are included as a multi-scale group.
            **kwargs:
                Additional arguments passed to :py:meth:`TiffPage.aszarr`
                or :py:meth:`TiffPageSeries.aszarr`.

        """
        if not self.pages:
            raise NotImplementedError('empty Zarr arrays not supported')
        if key is None and series is None:
            return self.series[0].aszarr(level=level, **kwargs)

        pages: Any
        if series is None:
            pages = self.pages
        else:
            if not isinstance(series, TiffPageSeries):
                series = self.series[series]
            if key is None:
                return series.aszarr(level=level, **kwargs)
            if level is not None:
                series = series.levels[level]
            pages = series

        if isinstance(key, (int, numpy.integer)):
            page: TiffPage | TiffFrame = pages[key]
            return page.aszarr(**kwargs)
        raise TypeError('key must be an integer index')

    @cached_property
    def series(self) -> list[TiffPageSeries]:
        """Series of pages with compatible shape and data type.

        Side effect: after accessing this property, `TiffFile.pages` might
        contain `TiffPage` and `TiffFrame` instead of only `TiffPage`
        instances.

        """
        if not self.pages:
            return []
        assert self.pages.keyframe is not None
        useframes = self.pages.useframes
        keyframe = self.pages.keyframe.index
        series: list[TiffPageSeries] | None = None
        for kind in (
            'shaped',
            'lsm',
            'mmstack',
            'ome',
            'imagej',
            'ndtiff',
            'fluoview',
            'stk',
            'sis',
            'svs',
            'scn',
            'qpi',
            'ndpi',
            'bif',
            'avs',
            'philips',
            'scanimage',
            # 'indica',  # TODO: rewrite _series_indica()
            'nih',
            'mdgel',  # adds second page to cache
            'uniform',
        ):
            if getattr(self, 'is_' + kind, False):
                series = getattr(self, '_series_' + kind)()
                if not series:
                    if kind == 'ome' and self.is_imagej:
                        # try ImageJ series if OME series fails.
                        # clear pages cache since _series_ome() might leave
                        # some frames without keyframe
                        self.pages._clear()
                        continue
                    if kind == 'mmstack':
                        # try OME, ImageJ, uniform
                        continue
                break
        if not series:
            series = self._series_generic()

        self.pages.useframes = useframes
        self.pages.set_keyframe(keyframe)

        # remove empty series, for example, in MD Gel files
        # series = [s for s in series if product(s.shape) > 0]
        assert series is not None
        for i, s in enumerate(series):
            s._index = i
        return series

    def _series_uniform(self) -> list[TiffPageSeries] | None:
        """Return all images in file as single series."""
        self.pages.useframes = True
        self.pages.set_keyframe(0)
        page = self.pages.first
        validate = not (page.is_scanimage or page.is_nih)
        pages = self.pages._getlist(validate=validate)
        if len(pages) == 1:
            shape = page.shape
            axes = page.axes
        else:
            shape = (len(pages),) + page.shape
            axes = 'I' + page.axes
        dtype = page.dtype
        return [TiffPageSeries(pages, shape, dtype, axes, kind='uniform')]

    def _series_generic(self) -> list[TiffPageSeries] | None:
        """Return image series in file.

        A series is a sequence of TiffPages with the same hash.

        """
        pages = self.pages
        pages._clear(False)
        pages.useframes = False
        if pages.cache:
            pages._load()

        series = []
        keys = []
        seriesdict: dict[int, list[TiffPage | TiffFrame]] = {}

        def addpage(page: TiffPage | TiffFrame, /) -> None:
            # add page to seriesdict
            if not page.shape:  # or product(page.shape) == 0:
                return
            key = page.hash
            if key in seriesdict:
                for p in seriesdict[key]:
                    if p.offset == page.offset:
                        break  # remove duplicate page
                else:
                    seriesdict[key].append(page)
            else:
                keys.append(key)
                seriesdict[key] = [page]

        for page in pages:
            addpage(page)
            if page.subifds is not None:
                for i, offset in enumerate(page.subifds):
                    if offset < 8:
                        continue
                    try:
                        self._fh.seek(offset)
                        subifd = TiffPage(self, (page.index, i))
                    except Exception as exc:
                        logger().warning(
                            f'{self!r} generic series raised {exc!r:.128}'
                        )
                    else:
                        addpage(subifd)

        for key in keys:
            pagelist = seriesdict[key]
            page = pagelist[0]
            shape = (len(pagelist),) + page.shape
            axes = 'I' + page.axes
            if 'S' not in axes:
                shape += (1,)
                axes += 'S'
            series.append(
                TiffPageSeries(
                    pagelist, shape, page.dtype, axes, kind='generic'
                )
            )

        self.is_uniform = len(series) == 1  # replaces is_uniform method
        if not self.is_agilent:
            pyramidize_series(series)
        return series

    def _series_shaped(self) -> list[TiffPageSeries] | None:
        """Return image series in tifffile "shaped" formatted file."""
        # TODO: all series need to have JSON metadata for this to succeed

        def append(
            series: list[TiffPageSeries],
            pages: list[TiffPage | TiffFrame | None],
            axes: str | None,
            shape: tuple[int, ...] | None,
            reshape: tuple[int, ...],
            name: str,
            truncated: bool | None,
        ) -> None:
            # append TiffPageSeries to series
            assert isinstance(pages[0], TiffPage)
            page = pages[0]
            if not check_shape(page.shape, reshape):
                logger().warning(
                    f'{self!r} shaped series metadata does not match '
                    f'page shape {page.shape} != {tuple(reshape)}'
                )
                failed = True
            else:
                failed = False
            if failed or axes is None or shape is None:
                shape = page.shape
                axes = page.axes
                if len(pages) > 1:
                    shape = (len(pages),) + shape
                    axes = 'Q' + axes
                if failed:
                    reshape = shape
            size = product(shape)
            resize = product(reshape)
            if page.is_contiguous and resize > size and resize % size == 0:
                if truncated is None:
                    truncated = True
                axes = 'Q' + axes
                shape = (resize // size,) + shape
            try:
                axes = reshape_axes(axes, shape, reshape)
                shape = reshape
            except ValueError as exc:
                logger().error(
                    f'{self!r} shaped series failed to reshape, '
                    f'raised {exc!r:.128}'
                )
            series.append(
                TiffPageSeries(
                    pages,
                    shape,
                    page.dtype,
                    axes,
                    name=name,
                    kind='shaped',
                    truncated=bool(truncated),
                    squeeze=False,
                )
            )

        def detect_series(
            pages: TiffPages | list[TiffPage | TiffFrame | None],
            series: list[TiffPageSeries]
        ) -> list[TiffPageSeries] | None:
            shape: tuple[int, ...] | None
            reshape: tuple[int, ...]
            page: TiffPage | TiffFrame | None
            keyframe: TiffPage
            subifds: list[TiffPage | TiffFrame | None] = []
            subifd: TiffPage | TiffFrame
            keysubifd: TiffPage
            axes: str | None
            name: str

            lenpages = len(pages)
            index = 0
            while True:
                if index >= lenpages:
                    break

                if isinstance(pages, TiffPages):
                    # new keyframe; start of new series
                    pages.set_keyframe(index)
                    keyframe = cast(TiffPage, pages.keyframe)
                else:
                    # pages is list of SubIFDs
                    keyframe = cast(TiffPage, pages[0])

                if keyframe.shaped_description is None:
                    logger().error(
                        f'{self!r} '
                        'invalid shaped series metadata or corrupted file'
                    )
                    return None
                # read metadata
                axes = None
                shape = None
                metadata = shaped_description_metadata(
                    keyframe.shaped_description
                )
                name = metadata.get('name', '')
                reshape = metadata['shape']
                truncated = None if keyframe.subifds is None else False
                truncated = metadata.get('truncated', truncated)
                if 'axes' in metadata:
                    axes = cast(str, metadata['axes'])
                    if len(axes) == len(reshape):
                        shape = reshape
                    else:
                        axes = ''
                        logger().error(
                            f'{self!r} shaped series axes do not match shape'
                        )
                # skip pages if possible
                spages: list[TiffPage | TiffFrame | None] = [keyframe]
                size = product(reshape)
                if size > 0:
                    npages, mod = divmod(size, product(keyframe.shape))
                else:
                    npages = 1
                    mod = 0
                if mod:
                    logger().error(
                        f'{self!r} '
                        'shaped series shape does not match page shape'
                    )
                    return None

                if 1 < npages <= lenpages - index:
                    assert keyframe._dtype is not None
                    size *= keyframe._dtype.itemsize
                    if truncated:
                        npages = 1
                    else:
                        page = pages[index + 1]
                        if (
                            keyframe.is_final
                            and page is not None
                            and keyframe.offset + size < page.offset
                            and keyframe.subifds is None
                        ):
                            truncated = False
                        else:
                            # must read all pages for series
                            truncated = False
                            for j in range(index + 1, index + npages):
                                page = pages[j]
                                assert page is not None
                                page.keyframe = keyframe
                                spages.append(page)
                append(series, spages, axes, shape, reshape, name, truncated)
                index += npages

                # create series from SubIFDs
                if keyframe.subifds:
                    subifds_size = len(keyframe.subifds)
                    for i, offset in enumerate(keyframe.subifds):
                        if offset < 8:
                            continue
                        subifds = []
                        for j, page in enumerate(spages):
                            # if page.subifds is not None:
                            try:
                                if (
                                    page is None
                                    or page.subifds is None
                                    or len(page.subifds) < subifds_size
                                ):
                                    raise ValueError(
                                        f'{page!r} contains invalid subifds'
                                    )
                                self._fh.seek(page.subifds[i])
                                if j == 0:
                                    subifd = TiffPage(self, (page.index, i))
                                    keysubifd = subifd
                                else:
                                    subifd = TiffFrame(
                                        self,
                                        (page.index, i),
                                        keyframe=keysubifd,
                                    )
                            except Exception as exc:
                                logger().error(
                                    f'{self!r} shaped series '
                                    f'raised {exc!r:.128}'
                                )
                                return None
                            subifds.append(subifd)
                        if subifds:
                            series_or_none = detect_series(subifds, series)
                            if series_or_none is None:
                                return None
                            series = series_or_none
            return series

        self.pages.useframes = True
        series = detect_series(self.pages, [])
        if series is None:
            return None
        self.is_uniform = len(series) == 1
        pyramidize_series(series, isreduced=True)
        return series

    def _series_imagej(self) -> list[TiffPageSeries] | None:
        """Return image series in ImageJ file."""
        # ImageJ's dimension order is TZCYXS
        # TODO: fix loading of color, composite, or palette images
        meta = self.imagej_metadata
        if meta is None:
            return None

        pages = self.pages
        pages.useframes = True
        pages.set_keyframe(0)
        page = self.pages.first

        order = meta.get('order', 'czt').lower()
        frames = meta.get('frames', 1)
        slices = meta.get('slices', 1)
        channels = meta.get('channels', 1)
        images = meta.get('images', 1)  # not reliable

        if images < 1 or frames < 1 or slices < 1 or channels < 1:
            logger().warning(
                f'{self!r} ImageJ series metadata invalid or corrupted file'
            )
            return None

        if channels == 1:
            images = frames * slices
        elif page.shaped[0] > 1 and page.shaped[0] == channels:
            # Bio-Formats declares separate samples as channels
            images = frames * slices
        elif images == frames * slices and page.shaped[4] == channels:
            # RGB contig samples declared as channel
            channels = 1
        else:
            images = frames * slices * channels

        if images == 1 and pages.is_multipage:
            images = len(pages)

        nbytes = images * page.nbytes

        # ImageJ virtual hyperstacks store all image metadata in the first
        # page and image data are stored contiguously before the second
        # page, if any
        if not page.is_final:
            isvirtual = False
        elif page.dataoffsets[0] + nbytes > self.filehandle.size:
            logger().error(
                f'{self!r} ImageJ series metadata invalid or corrupted file'
            )
            return None
        elif images <= 1:
            isvirtual = True
        elif (
            pages.is_multipage
            and page.dataoffsets[0] + nbytes > pages[1].offset
        ):
            # next page is not stored after data
            isvirtual = False
        else:
            isvirtual = True

        page_list: list[TiffPage | TiffFrame]
        if isvirtual:
            # no need to read other pages
            page_list = [page]
        else:
            page_list = pages[:]

        shape: tuple[int, ...]
        axes: str

        if order in {'czt', 'default'}:
            axes = 'TZC'
            shape = (frames, slices, channels)
        elif order == 'ctz':
            axes = 'ZTC'
            shape = (slices, frames, channels)
        elif order == 'zct':
            axes = 'TCZ'
            shape = (frames, channels, slices)
        elif order == 'ztc':
            axes = 'CTZ'
            shape = (channels, frames, slices)
        elif order == 'tcz':
            axes = 'ZCT'
            shape = (slices, channels, frames)
        elif order == 'tzc':
            axes = 'CZT'
            shape = (channels, slices, frames)
        else:
            axes = 'TZC'
            shape = (frames, slices, channels)
            logger().warning(
                f'{self!r} ImageJ series of unknown order {order!r}'
            )

        remain = images // product(shape)
        if remain > 1:
            logger().debug(
                f'{self!r} ImageJ series contains unidentified dimension'
            )
            shape = (remain,) + shape
            axes = 'I' + axes

        if page.shaped[0] > 1:
            # Bio-Formats declares separate samples as channels
            assert axes[-1] == 'C'
            shape = shape[:-1] + page.shape
            axes += page.axes[1:]
        else:
            shape += page.shape
            axes += page.axes

        if 'S' not in axes:
            shape += (1,)
            axes += 'S'
        # assert axes.endswith('TZCYXS'), axes

        truncated = (
            isvirtual and not pages.is_multipage and page.nbytes != nbytes
        )

        self.is_uniform = True
        return [
            TiffPageSeries(
                page_list,
                shape,
                page.dtype,
                axes,
                kind='imagej',
                truncated=truncated,
            )
        ]

    def _series_nih(self) -> list[TiffPageSeries] | None:
        """Return all images in NIH Image file as single series."""
        series = self._series_uniform()
        if series is not None:
            for s in series:
                s.kind = 'nih'
        return series

    def _series_scanimage(self) -> list[TiffPageSeries] | None:
        """Return image series in ScanImage file."""
        pages = self.pages._getlist(validate=False)
        page = self.pages.first
        dtype = page.dtype
        shape = None

        meta = self.scanimage_metadata
        if meta is None:
            framedata = {}
        else:
            framedata = meta.get('FrameData', {})
        if 'SI.hChannels.channelSave' in framedata:
            try:
                channels = framedata['SI.hChannels.channelSave']
                try:
                    # channelSave is a list of channel IDs
                    channels = len(channels)
                except TypeError:
                    # channelSave is a single channel ID
                    channels = 1
                # slices = framedata.get(
                #    'SI.hStackManager.actualNumSlices',
                #     framedata.get('SI.hStackManager.numSlices', None),
                # )
                # if slices is None:
                #     raise ValueError('unable to determine numSlices')
                slices = None
                try:
                    frames = int(framedata['SI.hStackManager.framesPerSlice'])
                except Exception as exc:
                    # framesPerSlice is inf
                    slices = 1
                    if len(pages) % channels:
                        raise ValueError(
                            'unable to determine framesPerSlice'
                        ) from exc
                    frames = len(pages) // channels
                if slices is None:
                    slices = max(len(pages) // (frames * channels), 1)
                shape = (slices, frames, channels) + page.shape
                axes = 'ZTC' + page.axes
            except Exception as exc:
                logger().warning(
                    f'{self!r} ScanImage series raised {exc!r:.128}'
                )

        # TODO: older versions of ScanImage store non-varying frame data in
        # the ImageDescription tag. Candidates are scanimage.SI5.channelsSave,
        # scanimage.SI5.stackNumSlices, scanimage.SI5.acqNumFrames
        # scanimage.SI4., state.acq.numberOfFrames, state.acq.numberOfFrames...

        if shape is None:
            shape = (len(pages),) + page.shape
            axes = 'I' + page.axes

        return [TiffPageSeries(pages, shape, dtype, axes, kind='scanimage')]

    def _series_fluoview(self) -> list[TiffPageSeries] | None:
        """Return image series in FluoView file."""
        meta = self.fluoview_metadata
        if meta is None:
            return None
        pages = self.pages._getlist(validate=False)
        mmhd = list(reversed(meta['Dimensions']))
        axes = ''.join(TIFF.MM_DIMENSIONS.get(i[0].upper(), 'Q') for i in mmhd)
        shape = tuple(int(i[1]) for i in mmhd)
        self.is_uniform = True
        return [
            TiffPageSeries(
                pages,
                shape,
                pages[0].dtype,
                axes,
                name=meta['ImageName'],
                kind='fluoview',
            )
        ]

    def _series_mdgel(self) -> list[TiffPageSeries] | None:
        """Return image series in MD Gel file."""
        # only a single page, scaled according to metadata in second page
        meta = self.mdgel_metadata
        if meta is None:
            return None
        transform: Callable[[NDArray[Any]], NDArray[Any]] | None
        self.pages.useframes = False
        self.pages.set_keyframe(0)

        if meta['FileTag'] in {2, 128}:
            dtype = numpy.dtype(numpy.float32)
            scale = meta['ScalePixel']
            scale = scale[0] / scale[1]  # rational
            if meta['FileTag'] == 2:
                # squary root data format
                def transform(a: NDArray[Any], /) -> NDArray[Any]:
                    return a.astype(numpy.float32) ** 2 * scale

            else:

                def transform(a: NDArray[Any], /) -> NDArray[Any]:
                    return a.astype(numpy.float32) * scale

        else:
            transform = None
        page = self.pages.first
        self.is_uniform = False
        return [
            TiffPageSeries(
                [page],
                page.shape,
                dtype,
                page.axes,
                transform=transform,
                kind='mdgel',
            )
        ]

    def _series_ndpi(self) -> list[TiffPageSeries] | None:
        """Return pyramidal image series in NDPI file."""
        series = self._series_generic()
        if series is None:
            return None
        for s in series:
            s.kind = 'ndpi'
            if s.axes[0] == 'I':
                s._set_dimensions(s.shape, 'Z' + s.axes[1:], None, True)
            if s.is_pyramidal:
                name = s.keyframe.tags.valueof(65427)
                s.name = 'Baseline' if name is None else name
                continue
            mag = s.keyframe.tags.valueof(65421)
            if mag is not None:
                if mag == -1.0:
                    s.name = 'Macro'
                    # s.kind += '_macro'
                elif mag == -2.0:
                    s.name = 'Map'
                    # s.kind += '_map'
        self.is_uniform = False
        return series

    def _series_avs(self) -> list[TiffPageSeries] | None:
        """Return pyramidal image series in AVS file."""
        series = self._series_generic()
        if series is None:
            return None
        if len(series) != 3:
            logger().warning(
                f'{self!r} AVS series expected 3 series, got {len(series)}'
            )
        s = series[0]
        s.kind = 'avs'
        if s.axes[0] == 'I':
            s._set_dimensions(s.shape, 'Z' + s.axes[1:], None, True)
        if s.is_pyramidal:
            s.name = 'Baseline'
        if len(series) == 3:
            series[1].name = 'Map'
            series[1].kind = 'avs'
            series[2].name = 'Macro'
            series[2].kind = 'avs'
        self.is_uniform = False
        return series

    def _series_philips(self) -> list[TiffPageSeries] | None:
        """Return pyramidal image series in Philips DP file."""
        from xml.etree import ElementTree as etree

        series = []
        pages = self.pages
        pages.cache = False
        pages.useframes = False
        pages.set_keyframe(0)
        pages._load()

        meta = self.philips_metadata
        assert meta is not None

        try:
            tree = etree.fromstring(meta)
        except etree.ParseError as exc:
            logger().error(f'{self!r} Philips series raised {exc!r:.128}')
            return None

        pixel_spacing = [
            tuple(float(v) for v in elem.text.replace('"', '').split())
            for elem in tree.findall(
                './/*'
                '/DataObject[@ObjectType="PixelDataRepresentation"]'
                '/Attribute[@Name="DICOM_PIXEL_SPACING"]'
            )
            if elem.text is not None
        ]
        if len(pixel_spacing) < 2:
            logger().error(
                f'{self!r} Philips series {len(pixel_spacing)=} < 2'
            )
            return None

        series_dict: dict[str, list[TiffPage]] = {}
        series_dict['Level'] = []
        series_dict['Other'] = []
        for page in pages:
            assert isinstance(page, TiffPage)
            if page.description.startswith('Macro'):
                series_dict['Macro'] = [page]
            elif page.description.startswith('Label'):
                series_dict['Label'] = [page]
            elif not page.is_tiled:
                series_dict['Other'].append(page)
            else:
                series_dict['Level'].append(page)

        levels = series_dict.pop('Level')
        if len(levels) != len(pixel_spacing):
            logger().error(
                f'{self!r} Philips series '
                f'{len(levels)=} != {len(pixel_spacing)=}'
            )
            return None

        # fix padding of sublevels
        imagewidth0 = levels[0].imagewidth
        imagelength0 = levels[0].imagelength
        h0, w0 = pixel_spacing[0]
        for serie, (h, w) in zip(levels[1:], pixel_spacing[1:]):
            page = serie.keyframe
            # if page.dtype.itemsize == 1:
            #     page.nodata = 255

            imagewidth = imagewidth0 // int(round(w / w0))
            imagelength = imagelength0 // int(round(h / h0))

            if page.imagewidth - page.tilewidth >= imagewidth:
                logger().warning(
                    f'{self!r} Philips series {page.index=} '
                    f'{page.imagewidth=}-{page.tilewidth=} >= {imagewidth=}'
                )
                page.imagewidth -= page.tilewidth - 1
            elif page.imagewidth < imagewidth:
                logger().warning(
                    f'{self!r} Philips series {page.index=} '
                    f'{page.imagewidth=} < {imagewidth=}'
                )
            else:
                page.imagewidth = imagewidth
            imagewidth = page.imagewidth

            if page.imagelength - page.tilelength >= imagelength:
                logger().warning(
                    f'{self!r} Philips series {page.index=} '
                    f'{page.imagelength=}-{page.tilelength=} >= {imagelength=}'
                )
                page.imagelength -= page.tilelength - 1
            # elif page.imagelength < imagelength:
            #    # in this case image is padded with zero
            else:
                page.imagelength = imagelength
            imagelength = page.imagelength

            if page.shaped[-1] > 1:
                page.shape = (imagelength, imagewidth, page.shape[-1])
            elif page.shaped[0] > 1:
                page.shape = (page.shape[0], imagelength, imagewidth)
            else:
                page.shape = (imagelength, imagewidth)
            page.shaped = (
                page.shaped[:2] + (imagelength, imagewidth) + page.shaped[-1:]
            )

        series = [TiffPageSeries([levels[0]], name='Baseline', kind='philips')]
        for i, page in enumerate(levels[1:]):
            series[0].levels.append(
                TiffPageSeries([page], name=f'Level{i + 1}', kind='philips')
            )
        for key, value in series_dict.items():
            for page in value:
                series.append(TiffPageSeries([page], name=key, kind='philips'))

        self.is_uniform = False
        return series

    def _series_indica(self) -> list[TiffPageSeries] | None:
        """Return pyramidal image series in IndicaLabs file."""
        # TODO: need more IndicaLabs sample files
        # TODO: parse indica series from XML
        # TODO: alpha channels in SubIFDs or main IFDs

        from xml.etree import ElementTree as etree

        series = self._series_generic()
        if series is None or len(series) != 1:
            return series

        try:
            tree = etree.fromstring(self.pages.first.description)
        except etree.ParseError as exc:
            logger().error(f'{self!r} Indica series raised {exc!r:.128}')
            return series

        channel_names = [
            channel.attrib['name'] for channel in tree.iter('channel')
        ]
        for s in series:
            s.kind = 'indica'
            # TODO: identify other dimensions
            if s.axes[0] == 'I' and s.shape[0] == len(channel_names):
                s._set_dimensions(s.shape, 'C' + s.axes[1:], None, True)
            if s.is_pyramidal:
                s.name = 'Baseline'
        self.is_uniform = False
        return series

    def _series_sis(self) -> list[TiffPageSeries] | None:
        """Return image series in Olympus SIS file."""
        meta = self.sis_metadata
        if meta is None:
            return None
        pages = self.pages._getlist(validate=False)  # TODO: this fails for VSI
        page = pages[0]
        lenpages = len(pages)

        if 'shape' in meta and 'axes' in meta:
            shape = meta['shape'] + page.shape
            axes = meta['axes'] + page.axes
        else:
            shape = (lenpages,) + page.shape
            axes = 'I' + page.axes
        self.is_uniform = True
        return [TiffPageSeries(pages, shape, page.dtype, axes, kind='sis')]

    def _series_qpi(self) -> list[TiffPageSeries] | None:
        """Return image series in PerkinElmer QPI file."""
        series = []
        pages = self.pages
        pages.cache = True
        pages.useframes = False
        pages.set_keyframe(0)
        pages._load()
        page0 = self.pages.first

        # Baseline
        # TODO: get name from ImageDescription XML
        ifds = []
        index = 0
        axes = 'C' + page0.axes
        dtype = page0.dtype
        pshape = page0.shape
        while index < len(pages):
            page = pages[index]
            if page.shape != pshape:
                break
            ifds.append(page)
            index += 1
        shape = (len(ifds),) + pshape
        series.append(
            TiffPageSeries(
                ifds, shape, dtype, axes, name='Baseline', kind='qpi'
            )
        )

        if index < len(pages):
            # Thumbnail
            page = pages[index]
            series.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='Thumbnail',
                    kind='qpi',
                )
            )
            index += 1

        if page0.is_tiled:
            # Resolutions
            while index < len(pages):
                pshape = (pshape[0] // 2, pshape[1] // 2) + pshape[2:]
                ifds = []
                while index < len(pages):
                    page = pages[index]
                    if page.shape != pshape:
                        break
                    ifds.append(page)
                    index += 1
                if len(ifds) != len(series[0].pages):
                    break
                shape = (len(ifds),) + pshape
                series[0].levels.append(
                    TiffPageSeries(
                        ifds, shape, dtype, axes, name='Resolution', kind='qpi'
                    )
                )

        if series[0].is_pyramidal and index < len(pages):
            # Macro
            page = pages[index]
            series.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='Macro',
                    kind='qpi',
                )
            )
            index += 1
            # Label
            if index < len(pages):
                page = pages[index]
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Label',
                        kind='qpi',
                    )
                )

        self.is_uniform = False
        return series

    def _series_svs(self) -> list[TiffPageSeries] | None:
        """Return image series in Aperio SVS file."""
        if not self.pages.first.is_tiled:
            return None

        series = []
        self.pages.cache = True
        self.pages.useframes = False
        self.pages.set_keyframe(0)
        self.pages._load()

        # baseline
        firstpage = self.pages.first
        if len(self.pages) == 1:
            self.is_uniform = False
            return [
                TiffPageSeries(
                    [firstpage],
                    firstpage.shape,
                    firstpage.dtype,
                    firstpage.axes,
                    name='Baseline',
                    kind='svs',
                )
            ]

        # thumbnail
        page = self.pages[1]
        thumnail = TiffPageSeries(
            [page],
            page.shape,
            page.dtype,
            page.axes,
            name='Thumbnail',
            kind='svs',
        )

        # resolutions and focal planes
        levels = {firstpage.shape: [firstpage]}
        index = 2
        while index < len(self.pages):
            page = cast(TiffPage, self.pages[index])
            if not page.is_tiled or page.is_reduced:
                break
            if page.shape in levels:
                levels[page.shape].append(page)
            else:
                levels[page.shape] = [page]
            index += 1

        zsize = len(levels[firstpage.shape])
        if not all(len(level) == zsize for level in levels.values()):
            logger().warning(f'{self!r} SVS series focal planes do not match')
            zsize = 1
        baseline = TiffPageSeries(
            levels[firstpage.shape],
            (zsize,) + firstpage.shape,
            firstpage.dtype,
            'Z' + firstpage.axes,
            name='Baseline',
            kind='svs',
        )
        for shape, level in levels.items():
            if shape == firstpage.shape:
                continue
            page = level[0]
            baseline.levels.append(
                TiffPageSeries(
                    level,
                    (zsize,) + page.shape,
                    page.dtype,
                    'Z' + page.axes,
                    name='Resolution',
                    kind='svs',
                )
            )
        series.append(baseline)
        series.append(thumnail)

        # Label, Macro; subfiletype 1, 9
        for _ in range(2):
            if index == len(self.pages):
                break
            page = self.pages[index]
            assert isinstance(page, TiffPage)
            if page.subfiletype == 9:
                name = 'Macro'
            else:
                name = 'Label'
            series.append(
                TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name=name,
                    kind='svs',
                )
            )
            index += 1
        self.is_uniform = False
        return series

    def _series_scn(self) -> list[TiffPageSeries] | None:
        """Return pyramidal image series in Leica SCN file."""
        # TODO: support collections
        from xml.etree import ElementTree as etree

        scnxml = self.pages.first.description
        root = etree.fromstring(scnxml)

        series = []
        self.pages.cache = True
        self.pages.useframes = False
        self.pages.set_keyframe(0)
        self.pages._load()

        for collection in root:
            if not collection.tag.endswith('collection'):
                continue
            for image in collection:
                if not image.tag.endswith('image'):
                    continue
                name = image.attrib.get('name', 'Unknown')
                for pixels in image:
                    if not pixels.tag.endswith('pixels'):
                        continue
                    resolutions: dict[int, dict[str, Any]] = {}
                    for dimension in pixels:
                        if not dimension.tag.endswith('dimension'):
                            continue
                        if int(image.attrib.get('sizeZ', 1)) > 1:
                            raise NotImplementedError(
                                'SCN series: Z-Stacks not supported. '
                                'Please submit a sample file.'
                            )
                        sizex = int(dimension.attrib['sizeX'])
                        sizey = int(dimension.attrib['sizeY'])
                        c = int(dimension.attrib.get('c', 0))
                        z = int(dimension.attrib.get('z', 0))
                        r = int(dimension.attrib.get('r', 0))
                        ifd = int(dimension.attrib['ifd'])
                        if r in resolutions:
                            level = resolutions[r]
                            if c > level['channels']:
                                level['channels'] = c
                            if z > level['sizez']:
                                level['sizez'] = z
                            level['ifds'][(c, z)] = ifd
                        else:
                            resolutions[r] = {
                                'size': [sizey, sizex],
                                'channels': c,
                                'sizez': z,
                                'ifds': {(c, z): ifd},
                            }
                    if not resolutions:
                        continue
                    levels = []
                    for r, level in sorted(resolutions.items()):
                        shape: tuple[int, ...] = (
                            level['channels'] + 1,
                            level['sizez'] + 1,
                        )
                        axes = 'CZ'

                        ifds: list[TiffPage | TiffFrame | None] = [
                            None
                        ] * product(shape)
                        for (c, z), ifd in sorted(level['ifds'].items()):
                            ifds[c * shape[1] + z] = self.pages[ifd]

                        assert ifds[0] is not None
                        axes += ifds[0].axes
                        shape += ifds[0].shape
                        dtype = ifds[0].dtype

                        levels.append(
                            TiffPageSeries(
                                ifds,
                                shape,
                                dtype,
                                axes,
                                parent=self,
                                name=name,
                                kind='scn',
                            )
                        )
                    levels[0].levels.extend(levels[1:])
                    series.append(levels[0])

        self.is_uniform = False
        return series

    def _series_bif(self) -> list[TiffPageSeries] | None:
        """Return image series in Ventana/Roche BIF file."""
        series = []
        baseline: TiffPageSeries | None = None
        self.pages.cache = True
        self.pages.useframes = False
        self.pages.set_keyframe(0)
        self.pages._load()

        for page in self.pages:
            page = cast(TiffPage, page)
            if page.description[:5] == 'Label':
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Label',
                        kind='bif',
                    )
                )
            elif (
                page.description == 'Thumbnail'
                or page.description[:11] == 'Probability'
            ):
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Thumbnail',
                        kind='bif',
                    )
                )
            elif 'level' not in page.description:
                # TODO: is this necessary?
                series.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Unknown',
                        kind='bif',
                    )
                )
            elif baseline is None:
                baseline = TiffPageSeries(
                    [page],
                    page.shape,
                    page.dtype,
                    page.axes,
                    name='Baseline',
                    kind='bif',
                )
                series.insert(0, baseline)
            else:
                baseline.levels.append(
                    TiffPageSeries(
                        [page],
                        page.shape,
                        page.dtype,
                        page.axes,
                        name='Resolution',
                        kind='bif',
                    )
                )

        logger().warning(f'{self!r} BIF series tiles are not stiched')
        self.is_uniform = False
        return series

    def _series_ome(self) -> list[TiffPageSeries] | None:
        """Return image series in OME-TIFF file(s)."""
        # xml.etree found to be faster than lxml
        from xml.etree import ElementTree as etree

        omexml = self.ome_metadata
        if omexml is None:
            return None
        try:
            root = etree.fromstring(omexml)
        except etree.ParseError as exc:
            # TODO: test badly encoded OME-XML
            logger().error(f'{self!r} OME series raised {exc!r:.128}')
            return None

        keyframe: TiffPage
        ifds: list[TiffPage | TiffFrame | None]
        size: int = -1

        def load_pages(tif: TiffFile, /) -> None:
            tif.pages.cache = True
            tif.pages.useframes = True
            tif.pages.set_keyframe(0)
            tif.pages._load(None)

        load_pages(self)

        root_uuid = root.attrib.get('UUID', None)
        self._files = {root_uuid: self}
        dirname = self._fh.dirname
        files_missing = 0
        moduloref = []
        modulo: dict[str, dict[str, tuple[str, int]]] = {}
        series: list[TiffPageSeries] = []
        for element in root:
            if element.tag.endswith('BinaryOnly'):
                # TODO: load OME-XML from master or companion file
                logger().debug(
                    f'{self!r} OME series is BinaryOnly, '
                    'not an OME-TIFF master file'
                )
                break
            if element.tag.endswith('StructuredAnnotations'):
                for annot in element:
                    if not annot.attrib.get('Namespace', '').endswith(
                        'modulo'
                    ):
                        continue
                    modulo[annot.attrib['ID']] = mod = {}
                    for value in annot:
                        for modulo_ns in value:
                            for along in modulo_ns:
                                if not along.tag[:-1].endswith('Along'):
                                    continue
                                axis = along.tag[-1]
                                newaxis = along.attrib.get('Type', 'other')
                                newaxis = TIFF.AXES_CODES[newaxis]
                                if 'Start' in along.attrib:
                                    step = float(along.attrib.get('Step', 1))
                                    start = float(along.attrib['Start'])
                                    stop = float(along.attrib['End']) + step
                                    labels = len(
                                        numpy.arange(start, stop, step)
                                    )
                                else:
                                    labels = len(
                                        [
                                            label
                                            for label in along
                                            if label.tag.endswith('Label')
                                        ]
                                    )
                                mod[axis] = (newaxis, labels)

            if not element.tag.endswith('Image'):
                continue

            for annot in element:
                if annot.tag.endswith('AnnotationRef'):
                    annotationref = annot.attrib['ID']
                    break
            else:
                annotationref = None

            attr = element.attrib
            name = attr.get('Name', None)

            for pixels in element:
                if not pixels.tag.endswith('Pixels'):
                    continue
                attr = pixels.attrib
                # dtype = attr.get('PixelType', None)
                axes = ''.join(reversed(attr['DimensionOrder']))
                shape = [int(attr['Size' + ax]) for ax in axes]
                ifds = []
                spp = 1  # samples per pixel
                first = True

                for data in pixels:
                    if data.tag.endswith('Channel'):
                        attr = data.attrib
                        if first:
                            first = False
                            spp = int(attr.get('SamplesPerPixel', spp))
                            if spp > 1:
                                # correct channel dimension for spp
                                shape = [
                                    shape[i] // spp if ax == 'C' else shape[i]
                                    for i, ax in enumerate(axes)
                                ]
                        elif int(attr.get('SamplesPerPixel', 1)) != spp:
                            raise ValueError(
                                'OME series cannot handle differing '
                                'SamplesPerPixel'
                            )
                        continue

                    if not data.tag.endswith('TiffData'):
                        continue

                    attr = data.attrib
                    ifd_index = int(attr.get('IFD', 0))
                    num = int(attr.get('NumPlanes', 1 if 'IFD' in attr else 0))
                    num = int(attr.get('PlaneCount', num))
                    idxs = [int(attr.get('First' + ax, 0)) for ax in axes[:-2]]
                    try:
                        idx = int(numpy.ravel_multi_index(idxs, shape[:-2]))
                    except ValueError as exc:
                        # ImageJ produces invalid ome-xml when cropping
                        logger().warning(
                            f'{self!r} '
                            'OME series contains invalid TiffData index, '
                            f'raised {exc!r:.128}',
                        )
                        continue
                    for uuid in data:
                        if not uuid.tag.endswith('UUID'):
                            continue
                        if (
                            root_uuid is None
                            and uuid.text is not None
                            and (
                                uuid.attrib.get('FileName', '').lower()
                                == self.filename.lower()
                            )
                        ):
                            # no global UUID, use this file
                            root_uuid = uuid.text
                            self._files[root_uuid] = self._files[None]
                            del self._files[None]
                        elif uuid.text not in self._files:
                            if not self._multifile:
                                # abort reading multifile OME series
                                # and fall back to generic series
                                return []
                            fname = uuid.attrib['FileName']
                            try:
                                if not self.filehandle.is_file:
                                    raise ValueError
                                tif = TiffFile(
                                    os.path.join(dirname, fname), _parent=self
                                )
                                load_pages(tif)
                            except (
                                OSError,
                                FileNotFoundError,
                                ValueError,
                            ) as exc:
                                if files_missing == 0:
                                    logger().warning(
                                        f'{self!r} OME series failed to read '
                                        f'{fname!r}, raised {exc!r:.128}. '
                                        'Missing data are zeroed'
                                    )
                                files_missing += 1
                                # assume that size is same as in previous file
                                # if no NumPlanes or PlaneCount are given
                                if num:
                                    size = num
                                elif size == -1:
                                    raise ValueError(
                                        'OME series missing '
                                        'NumPlanes or PlaneCount'
                                    ) from exc
                                ifds.extend([None] * (size + idx - len(ifds)))
                                break
                            self._files[uuid.text] = tif
                            tif.close()
                        pages = self._files[uuid.text].pages
                        try:
                            size = num if num else len(pages)
                            ifds.extend([None] * (size + idx - len(ifds)))
                            for i in range(size):
                                ifds[idx + i] = pages[ifd_index + i]
                        except IndexError as exc:
                            logger().warning(
                                f'{self!r} '
                                'OME series contains index out of range, '
                                f'raised {exc!r:.128}'
                            )
                        # only process first UUID
                        break
                    else:
                        # no uuid found
                        pages = self.pages
                        try:
                            size = num if num else len(pages)
                            ifds.extend([None] * (size + idx - len(ifds)))
                            for i in range(size):
                                ifds[idx + i] = pages[ifd_index + i]
                        except IndexError as exc:
                            logger().warning(
                                f'{self!r} '
                                'OME series contains index out of range, '
                                f'raised {exc!r:.128}'
                            )

                if not ifds or all(i is None for i in ifds):
                    # skip images without data
                    continue

                # find a keyframe
                for ifd in ifds:
                    # try find a TiffPage
                    if ifd is not None and ifd == ifd.keyframe:
                        keyframe = cast(TiffPage, ifd)
                        break
                else:
                    # reload a TiffPage from file
                    for i, ifd in enumerate(ifds):
                        if ifd is not None:
                            isclosed = ifd.parent.filehandle.closed
                            if isclosed:
                                ifd.parent.filehandle.open()
                            ifd.parent.pages.set_keyframe(ifd.index)
                            keyframe = cast(
                                TiffPage, ifd.parent.pages[ifd.index]
                            )
                            ifds[i] = keyframe
                            if isclosed:
                                keyframe.parent.filehandle.close()
                            break

                # does the series spawn multiple files
                multifile = False
                for ifd in ifds:
                    if ifd and ifd.parent != keyframe.parent:
                        multifile = True
                        break

                if spp > 1:
                    if keyframe.planarconfig == 1:
                        shape += [spp]
                        axes += 'S'
                    else:
                        shape = shape[:-2] + [spp] + shape[-2:]
                        axes = axes[:-2] + 'S' + axes[-2:]
                if 'S' not in axes:
                    shape += [1]
                    axes += 'S'

                # number of pages in the file might mismatch XML metadata, for
                # example Nikon-cell011.ome.tif or stack_t24_y2048_x2448.tiff
                size = max(product(shape) // keyframe.size, 1)
                if size < len(ifds):
                    logger().warning(
                        f'{self!r} '
                        f'OME series expected {size} frames, got {len(ifds)}'
                    )
                    ifds = ifds[:size]
                elif size > len(ifds):
                    logger().warning(
                        f'{self!r} '
                        f'OME series is missing {size - len(ifds)} frames.'
                        ' Missing data are zeroed'
                    )
                    ifds.extend([None] * (size - len(ifds)))

                # FIXME: this implementation assumes the last dimensions are
                # stored in TIFF pages. Apparently that is not always the case.
                # For example, TCX (20000, 2, 500) is stored in 2 pages of
                # (20000, 500) in 'Image 7.ome_h00.tiff'.
                # For now, verify that shapes of keyframe and series match.
                # If not, skip series.
                squeezed = squeeze_axes(shape, axes)[0]
                if keyframe.shape != tuple(squeezed[-len(keyframe.shape) :]):
                    logger().warning(
                        f'{self!r} OME series cannot handle discontiguous '
                        f'storage ({keyframe.shape} != '
                        f'{tuple(squeezed[-len(keyframe.shape) :])})',
                    )
                    del ifds
                    continue

                # set keyframe on all IFDs
                # each series must contain a TiffPage used as keyframe
                keyframes: dict[str, TiffPage] = {
                    keyframe.parent.filehandle.name: keyframe
                }
                for i, page in enumerate(ifds):
                    if page is None:
                        continue
                    fh = page.parent.filehandle
                    if fh.name not in keyframes:
                        if page.keyframe != page:
                            # reload TiffPage from file
                            isclosed = fh.closed
                            if isclosed:
                                fh.open()
                            page.parent.pages.set_keyframe(page.index)
                            page = page.parent.pages[page.index]
                            ifds[i] = page
                            if isclosed:
                                fh.close()
                        keyframes[fh.name] = cast(TiffPage, page)
                    if page.keyframe != page:
                        page.keyframe = keyframes[fh.name]

                moduloref.append(annotationref)
                series.append(
                    TiffPageSeries(
                        ifds,
                        shape,
                        keyframe.dtype,
                        axes,
                        parent=self,
                        name=name,
                        multifile=multifile,
                        kind='ome',
                    )
                )
                del ifds

        if files_missing > 1:
            logger().warning(
                f'{self!r} OME series failed to read {files_missing} files'
            )

        # apply modulo according to AnnotationRef
        for aseries, annotationref in zip(series, moduloref):
            if annotationref not in modulo:
                continue
            shape = list(aseries.get_shape(False))
            axes = aseries.get_axes(False)
            for axis, (newaxis, size) in modulo[annotationref].items():
                i = axes.index(axis)
                if shape[i] == size:
                    axes = axes.replace(axis, newaxis, 1)
                else:
                    shape[i] //= size
                    shape.insert(i + 1, size)
                    axes = axes.replace(axis, axis + newaxis, 1)
            aseries._set_dimensions(shape, axes, None)

        # pyramids
        for aseries in series:
            keyframe = aseries.keyframe
            if keyframe.subifds is None:
                continue
            if len(self._files) > 1:
                # TODO: support multi-file pyramids; must re-open/close
                logger().warning(
                    f'{self!r} OME series cannot read multi-file pyramids'
                )
                break
            for level in range(len(keyframe.subifds)):
                found_keyframe = False
                ifds = []
                for page in aseries.pages:
                    if (
                        page is None
                        or page.subifds is None
                        or page.subifds[level] < 8
                    ):
                        ifds.append(None)
                        continue
                    page.parent.filehandle.seek(page.subifds[level])
                    if page.keyframe == page:
                        ifd = keyframe = TiffPage(
                            self, (page.index, level + 1)
                        )
                        found_keyframe = True
                    elif not found_keyframe:
                        raise RuntimeError('no keyframe found')
                    else:
                        ifd = TiffFrame(
                            self, (page.index, level + 1), keyframe=keyframe
                        )
                    ifds.append(ifd)
                if all(ifd_or_none is None for ifd_or_none in ifds):
                    logger().warning(
                        f'{self!r} OME series level {level + 1} is empty'
                    )
                    break
                # fix shape
                shape = list(aseries.get_shape(False))
                axes = aseries.get_axes(False)
                for i, ax in enumerate(axes):
                    if ax == 'X':
                        shape[i] = keyframe.imagewidth
                    elif ax == 'Y':
                        shape[i] = keyframe.imagelength
                # add series
                aseries.levels.append(
                    TiffPageSeries(
                        ifds,
                        tuple(shape),
                        keyframe.dtype,
                        axes,
                        parent=self,
                        name=f'level {level + 1}',
                        kind='ome',
                    )
                )

        self.is_uniform = len(series) == 1 and len(series[0].levels) == 1

        return series

    def _series_mmstack(self) -> list[TiffPageSeries] | None:
        """Return series in Micro-Manager stack file(s)."""
        settings = self.micromanager_metadata
        if (
            settings is None
            or 'Summary' not in settings
            or 'IndexMap' not in settings
        ):
            return None

        pages: list[TiffPage | TiffFrame | None]
        page_count: int

        summary = settings['Summary']
        indexmap = settings['IndexMap']
        indexmap = indexmap[indexmap[:, 4].argsort()]

        if 'MicroManagerVersion' not in summary or 'Frames' not in summary:
            # TODO: handle MagellanStack?
            return None

        # determine CZTR shape from indexmap; TODO: is this necessary?
        indexmap_shape = (numpy.max(indexmap[:, :4], axis=0) + 1).tolist()
        indexmap_index = {'C': 0, 'Z': 1, 'T': 2, 'R': 3}

        # TODO: activate this?
        # if 'AxisOrder' in summary:
        #     axesorder = summary['AxisOrder']
        #     keys = {
        #         'channel': 'C',
        #         'z': 'Z',
        #         'slice': 'Z',
        #         'position': 'R',
        #         'time': 'T',
        #     }
        #     axes = ''.join(keys[ax] for ax in reversed(axesorder))

        axes = 'TR' if summary.get('TimeFirst', True) else 'RT'
        axes += 'ZC' if summary.get('SlicesFirst', True) else 'CZ'

        keys = {
            'C': 'Channels',
            'Z': 'Slices',
            'R': 'Positions',
            'T': 'Frames',
        }
        shape = tuple(
            max(
                indexmap_shape[indexmap_index[ax]],
                int(summary.get(keys[ax], 1)),
            )
            for ax in axes
        )
        size = product(shape)

        indexmap_order = tuple(indexmap_index[ax] for ax in axes)

        def add_file(tif: TiffFile, indexmap: NDArray[Any]) -> int:
            # add virtual TiffFrames to pages list
            page_count = 0
            offsets: list[int]
            offsets = indexmap[:, 4].tolist()  # type: ignore[assignment]
            indices = numpy.ravel_multi_index(
                # type: ignore[call-overload]
                indexmap[:, indexmap_order].T,
                shape,
            ).tolist()
            keyframe = tif.pages.first
            filesize = tif.filehandle.size - keyframe.databytecounts[0] - 162
            index: int
            offset: int
            for index, offset in zip(indices, offsets):
                if offset == keyframe.offset:
                    pages[index] = keyframe
                    page_count += 1
                    continue
                if 0 < offset <= filesize:
                    dataoffsets = (offset + 162,)
                    databytecounts = keyframe.databytecounts
                    page_count += 1
                else:
                    # assume file is truncated
                    dataoffsets = databytecounts = (0,)
                    offset = 0
                pages[index] = TiffFrame(
                    tif,
                    index=index,
                    offset=offset,
                    dataoffsets=dataoffsets,
                    databytecounts=databytecounts,
                    keyframe=keyframe,
                )
            return page_count

        multifile = size > indexmap.shape[0]
        if multifile:
            # get multifile prefix
            if not self.filehandle.is_file:
                logger().warning(
                    f'{self!r} MMStack multi-file series cannot be read from '
                    f'{self.filehandle._fh!r}'
                )
                multifile = False
            elif '_MMStack' not in self.filename:
                logger().warning(f'{self!r} MMStack file name is invalid')
                multifile = False
            elif 'Prefix' in summary:
                prefix = summary['Prefix']
                if not self.filename.startswith(prefix):
                    logger().warning(f'{self!r} MMStack file name is invalid')
                    multifile = False
            else:
                prefix = self.filename.split('_MMStack')[0]

        if multifile:
            # read other files
            pattern = os.path.join(
                self.filehandle.dirname, prefix + '_MMStack*.tif'
            )
            filenames = glob.glob(pattern)
            if len(filenames) == 1:
                multifile = False
            else:
                pages = [None] * size
                page_count = add_file(self, indexmap)
                for fname in filenames:
                    if self.filename == os.path.split(fname)[-1]:
                        continue
                    with TiffFile(fname) as tif:
                        indexmap = read_micromanager_metadata(
                            tif.filehandle, {'IndexMap'}
                        )['IndexMap']
                        indexmap = indexmap[indexmap[:, 4].argsort()]
                        page_count += add_file(tif, indexmap)

        if multifile:
            pass
        elif size > indexmap.shape[0]:
            # other files missing: squeeze shape
            old_shape = shape
            min_index = numpy.min(indexmap[:, :4], axis=0)
            max_index = numpy.max(indexmap[:, :4], axis=0)
            indexmap = indexmap.copy()
            indexmap[:, :4] -= min_index
            shape = tuple(
                j - i + 1
                for i, j in zip(min_index.tolist(), max_index.tolist())
            )
            shape = tuple(shape[i] for i in indexmap_order)
            size = product(shape)
            pages = [None] * size
            page_count = add_file(self, indexmap)
            logger().warning(
                f'{self!r} MMStack series is missing files. '
                f'Returning subset {shape!r} of {old_shape!r}'
            )
        else:
            # single file
            pages = [None] * size
            page_count = add_file(self, indexmap)

        if page_count != size:
            logger().warning(
                f'{self!r} MMStack is missing {size - page_count} pages.'
                ' Missing data are zeroed'
            )

        keyframe = self.pages.first
        return [
            TiffPageSeries(
                pages,
                shape=shape + keyframe.shape,
                dtype=keyframe.dtype,
                axes=axes + keyframe.axes,
                # axestiled=axestiled,
                # axesoverlap=axesoverlap,
                # coords=coords,
                parent=self,
                kind='mmstack',
                multifile=multifile,
                squeeze=True,
            )
        ]

    def _series_ndtiff(self) -> list[TiffPageSeries] | None:
        """Return series in NDTiff v2 and v3 files."""
        # TODO: implement fallback for missing index file, versions 0 and 1
        if not self.filehandle.is_file:
            logger().warning(
                f'{self!r} NDTiff.index not found for {self.filehandle._fh!r}'
            )
            return None

        indexfile = os.path.join(self.filehandle.dirname, 'NDTiff.index')
        if not os.path.exists(indexfile):
            logger().warning(f'{self!r} NDTiff.index not found')
            return None

        keyframes: dict[str, TiffPage] = {}
        shape: tuple[int, ...]
        dims: tuple[str, ...]
        page: TiffPage | TiffFrame
        pageindex = 0
        pixel_types = {
            0: ('uint8', 8),  # 8bit monochrome
            1: ('uint16', 16),  # 16bit monochrome
            2: ('uint8', 8),  # 8bit RGB
            3: ('uint16', 10),  # 10bit monochrome
            4: ('uint16', 12),  # 12bit monochrome
            5: ('uint16', 14),  # 14bit monochrome
            6: ('uint16', 11),  # 11bit monochrome
        }

        indices: dict[tuple[int, ...], TiffPage | TiffFrame] = {}
        categories: dict[str, dict[str, int]] = {}
        first = True

        for (
            axes_dict,
            filename,
            dataoffset,
            width,
            height,
            pixeltype,
            compression,
            metaoffset,
            metabytecount,
            metacompression,
        ) in read_ndtiff_index(indexfile):
            if filename in keyframes:
                # create virtual frame from index
                pageindex += 1  # TODO
                keyframe = keyframes[filename]
                page = TiffFrame(
                    keyframe.parent,
                    pageindex,
                    offset=None,  # virtual frame
                    keyframe=keyframe,
                    dataoffsets=(dataoffset,),
                    databytecounts=keyframe.databytecounts,
                )
                if page.shape[:2] != (height, width):
                    raise ValueError(
                        'NDTiff.index does not match TIFF shape '
                        f'{page.shape[:2]} != {(height, width)}'
                    )
                if compression != 0:
                    raise ValueError(
                        'NDTiff.index compression {compression} not supported'
                    )
                if page.compression != 1:
                    raise ValueError(
                        'NDTiff.index does not match TIFF compression '
                        f'{page.compression!r}'
                    )
                if pixeltype not in pixel_types:
                    raise ValueError(
                        f'NDTiff.index unknown pixel type {pixeltype}'
                    )
                dtype, _ = pixel_types[pixeltype]
                if page.dtype != dtype:
                    raise ValueError(
                        'NDTiff.index pixeltype does not match TIFF dtype '
                        f'{page.dtype} != {dtype}'
                    )
            elif filename == self.filename:
                # use first page as keyframe
                pageindex = 0
                page = self.pages.first
                keyframes[filename] = page
            else:
                # read keyframe from file
                pageindex = 0
                with TiffFile(
                    os.path.join(self.filehandle.dirname, filename)
                ) as tif:
                    page = tif.pages.first
                keyframes[filename] = page

            # replace string with integer indices
            index: int | str
            if first:
                for axis, index in axes_dict.items():
                    if isinstance(index, str):
                        categories[axis] = {index: 0}
                        axes_dict[axis] = 0
                first = False
            elif categories:
                for axis, values in categories.items():
                    index = axes_dict[axis]
                    assert isinstance(index, str)
                    if index not in values:
                        values[index] = max(values.values()) + 1
                    axes_dict[axis] = values[index]

            indices[tuple(axes_dict.values())] = page  # type: ignore[arg-type]
            dims = tuple(axes_dict.keys())

        # indices may be negative or missing
        indices_array = numpy.array(list(indices.keys()), dtype=numpy.int32)
        min_index = numpy.min(indices_array, axis=0).tolist()
        max_index = numpy.max(indices_array, axis=0).tolist()
        shape = tuple(j - i + 1 for i, j in zip(min_index, max_index))

        # change axes to match storage order
        order = order_axes(indices_array, squeeze=False)
        shape = tuple(shape[i] for i in order)
        dims = tuple(dims[i] for i in order)
        indices = {
            tuple(index[i] - min_index[i] for i in order): value
            for index, value in indices.items()
        }

        pages: list[TiffPage | TiffFrame | None] = []
        for idx in numpy.ndindex(shape):
            pages.append(indices.get(idx, None))

        keyframe = next(i for i in keyframes.values())
        shape += keyframe.shape
        dims += keyframe.dims
        axes = ''.join(TIFF.AXES_CODES.get(i.lower(), 'Q') for i in dims)

        # TODO: support tiled axes and overlap
        # meta: Any = self.micromanager_metadata
        # if meta is None:
        #     meta = {}
        # elif 'Summary' in meta:
        #     meta = meta['Summary']

        # # map axes column->x, row->y
        # axestiled: dict[int, int] = {}
        # axesoverlap: dict[int, int] = {}
        # if 'column' in dims:
        #     key = dims.index('column')
        #     axestiled[key] = keyframe.axes.index('X')
        #     axesoverlap[key] = meta.get('GridPixelOverlapX', 0)
        # if 'row' in dims:
        #     key = dims.index('row')
        #     axestiled[key] = keyframe.axes.index('Y')
        #     axesoverlap[key] = meta.get('GridPixelOverlapY', 0)

        # if all(i == 0 for i in axesoverlap.values()):
        #     axesoverlap = {}

        self.is_uniform = True
        return [
            TiffPageSeries(
                pages,
                shape=shape,
                dtype=keyframe.dtype,
                axes=axes,
                # axestiled=axestiled,
                # axesoverlap=axesoverlap,
                # coords=coords,
                parent=self,
                kind='ndtiff',
                multifile=len(keyframes) > 1,
                squeeze=True,
            )
        ]

    def _series_stk(self) -> list[TiffPageSeries] | None:
        """Return series in STK file."""
        meta = self.stk_metadata
        if meta is None:
            return None
        page = self.pages.first
        planes = meta['NumberPlanes']
        name = meta.get('Name', '')
        if planes == 1:
            shape = (1,) + page.shape
            axes = 'I' + page.axes
        elif numpy.all(meta['ZDistance'] != 0):
            shape = (planes,) + page.shape
            axes = 'Z' + page.axes
        elif numpy.all(numpy.diff(meta['TimeCreated']) != 0):
            shape = (planes,) + page.shape
            axes = 'T' + page.axes
        else:
            # TODO: determine other/combinations of dimensions
            shape = (planes,) + page.shape
            axes = 'I' + page.axes
        self.is_uniform = True
        series = TiffPageSeries(
            [page],
            shape,
            page.dtype,
            axes,
            name=name,
            truncated=planes > 1,
            kind='stk',
        )
        return [series]

    def _series_lsm(self) -> list[TiffPageSeries] | None:
        """Return main and thumbnail series in LSM file."""
        lsmi = self.lsm_metadata
        if lsmi is None:
            return None
        axes = TIFF.CZ_LSMINFO_SCANTYPE[lsmi['ScanType']]
        if self.pages.first.planarconfig == 1:
            axes = axes.replace('C', '').replace('X', 'XC')
        elif self.pages.first.planarconfig == 2:
            # keep axis for `get_shape(False)`
            pass
        elif self.pages.first.samplesperpixel == 1:
            axes = axes.replace('C', '')
        if lsmi.get('DimensionP', 0) > 0:
            axes = 'P' + axes
        if lsmi.get('DimensionM', 0) > 0:
            axes = 'M' + axes
        shape = tuple(int(lsmi[TIFF.CZ_LSMINFO_DIMENSIONS[i]]) for i in axes)

        name = lsmi.get('Name', '')
        pages = self.pages._getlist(slice(0, None, 2), validate=False)
        dtype = pages[0].dtype
        series = [
            TiffPageSeries(pages, shape, dtype, axes, name=name, kind='lsm')
        ]

        page = cast(TiffPage, self.pages[1])
        if page.is_reduced:
            pages = self.pages._getlist(slice(1, None, 2), validate=False)
            dtype = page.dtype
            cp = 1
            i = 0
            while cp < len(pages) and i < len(shape) - 2:
                cp *= shape[i]
                i += 1
            shape = shape[:i] + page.shape
            axes = axes[:i] + page.axes
            series.append(
                TiffPageSeries(
                    pages, shape, dtype, axes, name=name, kind='lsm'
                )
            )

        self.is_uniform = False
        return series

    def _lsm_load_pages(self) -> None:
        """Read and fix all pages from LSM file."""
        # cache all pages to preserve corrected values
        pages = self.pages
        pages.cache = True
        pages.useframes = True
        # use first and second page as keyframes
        pages.set_keyframe(1)
        pages.set_keyframe(0)
        # load remaining pages as frames
        pages._load(None)
        # fix offsets and bytecounts first
        # TODO: fix multiple conversions between lists and tuples
        self._lsm_fix_strip_offsets()
        self._lsm_fix_strip_bytecounts()
        # assign keyframes for data and thumbnail series
        keyframe = self.pages.first
        for page in pages._pages[::2]:
            page.keyframe = keyframe  # type: ignore[union-attr]
        keyframe = cast(TiffPage, pages[1])
        for page in pages._pages[1::2]:
            page.keyframe = keyframe  # type: ignore[union-attr]

    def _lsm_fix_strip_offsets(self) -> None:
        """Unwrap strip offsets for LSM files greater than 4 GB.

        Each series and position require separate unwrapping (undocumented).

        """
        if self.filehandle.size < 2**32:
            return

        indices: NDArray[Any]
        pages = self.pages
        npages = len(pages)
        series = self.series[0]
        axes = series.axes

        # find positions
        positions = 1
        for i in 0, 1:
            if series.axes[i] in 'PM':
                positions *= series.shape[i]

        # make time axis first
        if positions > 1:
            ntimes = 0
            for i in 1, 2:
                if axes[i] == 'T':
                    ntimes = series.shape[i]
                    break
            if ntimes:
                div, mod = divmod(npages, 2 * positions * ntimes)
                if mod != 0:
                    raise RuntimeError('mod != 0')
                shape = (positions, ntimes, div, 2)
                indices = numpy.arange(product(shape)).reshape(shape)
                indices = numpy.moveaxis(indices, 1, 0)
            else:
                indices = numpy.arange(npages).reshape(-1, 2)
        else:
            indices = numpy.arange(npages).reshape(-1, 2)

        # images of reduced page might be stored first
        if pages[0].dataoffsets[0] > pages[1].dataoffsets[0]:
            indices = indices[..., ::-1]

        # unwrap offsets
        wrap = 0
        previousoffset = 0
        for npi in indices.flat:
            page = pages[int(npi)]
            dataoffsets = []
            if all(i <= 0 for i in page.dataoffsets):
                logger().warning(
                    f'{self!r} LSM file incompletely written at {page}'
                )
                break
            for currentoffset in page.dataoffsets:
                if currentoffset < previousoffset:
                    wrap += 2**32
                dataoffsets.append(currentoffset + wrap)
                previousoffset = currentoffset
            page.dataoffsets = tuple(dataoffsets)

    def _lsm_fix_strip_bytecounts(self) -> None:
        """Set databytecounts to size of compressed data.

        The StripByteCounts tag in LSM files contains the number of bytes
        for the uncompressed data.

        """
        if self.pages.first.compression == 1:
            return
        # sort pages by first strip offset
        pages = sorted(self.pages, key=lambda p: p.dataoffsets[0])
        npages = len(pages) - 1
        for i, page in enumerate(pages):
            if page.index % 2:
                continue
            offsets = page.dataoffsets
            bytecounts = page.databytecounts
            if i < npages:
                lastoffset = pages[i + 1].dataoffsets[0]
            else:
                # LZW compressed strips might be longer than uncompressed
                lastoffset = min(
                    offsets[-1] + 2 * bytecounts[-1], self._fh.size
                )
            bytecount_list = list(bytecounts)
            for j in range(len(bytecounts) - 1):
                bytecount_list[j] = offsets[j + 1] - offsets[j]
            bytecount_list[-1] = lastoffset - offsets[-1]
            page.databytecounts = tuple(bytecount_list)

    def _ndpi_load_pages(self) -> None:
        """Read and fix pages from NDPI slide file if CaptureMode > 6.

        If the value of the CaptureMode tag is greater than 6, change the
        attributes of TiffPage instances that are part of the pyramid to
        match 16-bit grayscale data. TiffTag values are not corrected.

        """
        pages = self.pages
        capturemode = self.pages.first.tags.valueof(65441)
        if capturemode is None or capturemode < 6:
            return

        pages.cache = True
        pages.useframes = False
        pages._load()

        for page in pages:
            assert isinstance(page, TiffPage)
            mag = page.tags.valueof(65421)
            if mag is None or mag > 0:
                page.photometric = PHOTOMETRIC.MINISBLACK
                page.sampleformat = SAMPLEFORMAT.UINT
                page.samplesperpixel = 1
                page.bitspersample = 16
                page.dtype = page._dtype = numpy.dtype(numpy.uint16)
                if page.shaped[-1] > 1:
                    page.axes = page.axes[:-1]
                    page.shape = page.shape[:-1]
                    page.shaped = page.shaped[:-1] + (1,)

    def __getattr__(self, name: str, /) -> bool:
        """Return `is_flag` attributes from first page."""
        if name[3:] in TIFF.PAGE_FLAGS:
            if not self.pages:
                return False
            value = bool(getattr(self.pages.first, name))
            setattr(self, name, value)
            return value
        raise AttributeError(
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
        )

    def __enter__(self) -> TiffFile:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<tifffile.TiffFile {snipstr(self._fh.name, 32)!r}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string containing information about TiffFile.

        The `detail` parameter specifies the level of detail returned:

        0: file only.
        1: all series, first page of series and its tags.
        2: large tag values and file metadata.
        3: all pages.

        """
        info_list = [
            "TiffFile '{}'",
            format_size(self._fh.size),
            (
                ''
                if byteorder_isnative(self.byteorder)
                else {'<': 'little-endian', '>': 'big-endian'}[self.byteorder]
            ),
        ]
        if self.is_bigtiff:
            info_list.append('BigTiff')
        if len(self.pages) > 1:
            info_list.append(f'{len(self.pages)} Pages')
        if len(self.series) > 1:
            info_list.append(f'{len(self.series)} Series')
        if len(self._files) > 1:
            info_list.append(f'{len(self._files)} Files')
        flags = self.flags
        if 'uniform' in flags and len(self.pages) == 1:
            flags.discard('uniform')
        info_list.append('|'.join(f.lower() for f in sorted(flags)))
        info = '  '.join(info_list)
        info = info.replace('    ', '  ').replace('   ', '  ')
        info = info.format(
            snipstr(self._fh.name, max(12, width + 2 - len(info)))
        )
        if detail <= 0:
            return info
        info_list = [info]
        info_list.append('\n'.join(str(s) for s in self.series))
        if detail >= 3:
            for page in self.pages:
                if page is None:
                    continue
                info_list.append(page._str(detail=detail, width=width))
                if page.pages is not None:
                    for subifd in page.pages:
                        info_list.append(
                            subifd._str(detail=detail, width=width)
                        )
        elif self.series:
            info_list.extend(
                s.keyframe._str(detail=detail, width=width)
                for s in self.series
                if not s.keyframe.parent.filehandle.closed  # avoid warning
            )
        elif self.pages:  # and self.pages.first:
            info_list.append(self.pages.first._str(detail=detail, width=width))
        if detail >= 2:
            for name in sorted(self.flags):
                if hasattr(self, name + '_metadata'):
                    m = getattr(self, name + '_metadata')
                    if m:
                        info_list.append(
                            f'{name.upper()}_METADATA\n'
                            f'{pformat(m, width=width, height=detail * 24)}'
                        )
        return '\n\n'.join(info_list).replace('\n\n\n', '\n\n')

    @cached_property
    def flags(self) -> set[str]:
        """Set of file flags (a potentially expensive operation)."""
        return {
            name.lower()
            for name in TIFF.FILE_FLAGS
            if getattr(self, 'is_' + name)
        }

    @cached_property
    def is_uniform(self) -> bool:
        """File contains uniform series of pages."""
        # the hashes of IFDs 0, 7, and -1 are the same
        pages = self.pages
        try:
            page = self.pages.first
        except IndexError:
            return False
        if page.subifds:
            return False
        if page.is_scanimage or page.is_nih:
            return True
        i = 0
        useframes = pages.useframes
        try:
            pages.useframes = False
            h = page.hash
            for i in (1, 7, -1):
                if pages[i].aspage().hash != h:
                    return False
        except IndexError:
            return i == 1  # single page TIFF is uniform
        finally:
            pages.useframes = useframes
        return True

    @property
    def is_appendable(self) -> bool:
        """Pages can be appended to file without corrupting."""
        # TODO: check other formats
        return not (
            self.is_ome
            or self.is_lsm
            or self.is_stk
            or self.is_imagej
            or self.is_fluoview
            or self.is_micromanager
        )

    @property
    def is_bigtiff(self) -> bool:
        """File has BigTIFF format."""
        return self.tiff.is_bigtiff

    @cached_property
    def is_ndtiff(self) -> bool:
        """File has NDTiff format."""
        # file should be accompanied by NDTiff.index
        meta = self.micromanager_metadata
        if meta is not None and meta.get('MajorVersion', 0) >= 2:
            self.is_uniform = True
            return True
        return False

    @cached_property
    def is_mmstack(self) -> bool:
        """File has Micro-Manager stack format."""
        meta = self.micromanager_metadata
        if (
            meta is not None
            and 'Summary' in meta
            and 'IndexMap' in meta
            and meta.get('MajorVersion', 1) == 0
            # and 'MagellanStack' not in self.filename:
        ):
            self.is_uniform = True
            return True
        return False

    @cached_property
    def is_mdgel(self) -> bool:
        """File has MD Gel format."""
        # side effect: add second page, if exists, to cache
        try:
            ismdgel = (
                self.pages.first.is_mdgel
                or self.pages.get(1, cache=True).is_mdgel
            )
            if ismdgel:
                self.is_uniform = False
            return ismdgel
        except IndexError:
            return False

    @property
    def is_sis(self) -> bool:
        """File is Olympus SIS format."""
        try:
            return (
                self.pages.first.is_sis
                and not self.filename.lower().endswith('.vsi')
            )
        except IndexError:
            return False

    @cached_property
    def shaped_metadata(self) -> tuple[dict[str, Any], ...] | None:
        """Tifffile metadata from JSON formatted ImageDescription tags."""
        if not self.is_shaped:
            return None
        result = []
        for s in self.series:
            if s.kind.lower() != 'shaped':
                continue
            page = s.pages[0]
            if (
                not isinstance(page, TiffPage)
                or page.shaped_description is None
            ):
                continue
            result.append(shaped_description_metadata(page.shaped_description))
        return tuple(result)

    @property
    def ome_metadata(self) -> str | None:
        """OME XML metadata from ImageDescription tag."""
        if not self.is_ome:
            return None
        # return xml2dict(self.pages.first.description)['OME']
        if self._omexml:
            return self._omexml
        return self.pages.first.description

    @property
    def scn_metadata(self) -> str | None:
        """Leica SCN XML metadata from ImageDescription tag."""
        if not self.is_scn:
            return None
        return self.pages.first.description

    @property
    def philips_metadata(self) -> str | None:
        """Philips DP XML metadata from ImageDescription tag."""
        if not self.is_philips:
            return None
        return self.pages.first.description

    @property
    def indica_metadata(self) -> str | None:
        """IndicaLabs XML metadata from ImageDescription tag."""
        if not self.is_indica:
            return None
        return self.pages.first.description

    @property
    def avs_metadata(self) -> str | None:
        """Argos AVS XML metadata from tag 65000."""
        if not self.is_avs:
            return None
        return self.pages.first.tags.valueof(65000)

    @property
    def lsm_metadata(self) -> dict[str, Any] | None:
        """LSM metadata from CZ_LSMINFO tag."""
        if not self.is_lsm:
            return None
        return self.pages.first.tags.valueof(34412)  # CZ_LSMINFO

    @cached_property
    def stk_metadata(self) -> dict[str, Any] | None:
        """STK metadata from UIC tags."""
        if not self.is_stk:
            return None
        page = self.pages.first
        tags = page.tags
        result: dict[str, Any] = {}
        if page.description:
            result['PlaneDescriptions'] = page.description.split('\x00')
        tag = tags.get(33629)  # UIC2tag
        result['NumberPlanes'] = 1 if tag is None else tag.count
        value = tags.valueof(33628)  # UIC1tag
        if value is not None:
            result.update(value)
        value = tags.valueof(33630)  # UIC3tag
        if value is not None:
            result.update(value)  # wavelengths
        value = tags.valueof(33631)  # UIC4tag
        if value is not None:
            result.update(value)  # override UIC1 tags
        uic2tag = tags.valueof(33629)
        if uic2tag is not None:
            result['ZDistance'] = uic2tag['ZDistance']
            result['TimeCreated'] = uic2tag['TimeCreated']
            result['TimeModified'] = uic2tag['TimeModified']
            for key in ('Created', 'Modified'):
                try:
                    result['Datetime' + key] = numpy.array(
                        [
                            julian_datetime(*dt)
                            for dt in zip(
                                uic2tag['Date' + key], uic2tag['Time' + key]
                            )
                        ],
                        dtype='datetime64[ns]',
                    )
                except Exception as exc:
                    result['Datetime' + key] = None
                    logger().warning(
                        f'{self!r} STK Datetime{key} raised {exc!r:.128}'
                    )
        return result

    @cached_property
    def imagej_metadata(self) -> dict[str, Any] | None:
        """ImageJ metadata from ImageDescription and IJMetadata tags."""
        if not self.is_imagej:
            return None
        page = self.pages.first
        if page.imagej_description is None:
            return None
        result = imagej_description_metadata(page.imagej_description)
        value = page.tags.valueof(50839)  # IJMetadata
        if value is not None:
            try:
                result.update(value)
            except Exception:
                pass
        return result

    @cached_property
    def fluoview_metadata(self) -> dict[str, Any] | None:
        """FluoView metadata from MM_Header and MM_Stamp tags."""
        if not self.is_fluoview:
            return None
        result = {}
        page = self.pages.first
        value = page.tags.valueof(34361)  # MM_Header
        if value is not None:
            result.update(value)
        # TODO: read stamps from all pages
        value = page.tags.valueof(34362)  # MM_Stamp
        if value is not None:
            result['Stamp'] = value
        # skip parsing image description; not reliable
        # try:
        #     t = fluoview_description_metadata(page.image_description)
        #     if t is not None:
        #         result['ImageDescription'] = t
        # except Exception as exc:
        #     logger().warning(
        #         f'{self!r} <fluoview_description_metadata> '
        #         f'raised {exc!r:.128}'
        #     )
        return result

    @property
    def nih_metadata(self) -> dict[str, Any] | None:
        """NIHImage metadata from NIHImageHeader tag."""
        if not self.is_nih:
            return None
        return self.pages.first.tags.valueof(43314)  # NIHImageHeader

    @property
    def fei_metadata(self) -> dict[str, Any] | None:
        """FEI metadata from SFEG or HELIOS tags."""
        if not self.is_fei:
            return None
        tags = self.pages.first.tags
        result = {}
        try:
            result.update(tags.valueof(34680))  # FEI_SFEG
        except Exception:
            pass
        try:
            result.update(tags.valueof(34682))  # FEI_HELIOS
        except Exception:
            pass
        return result

    @property
    def sem_metadata(self) -> dict[str, Any] | None:
        """SEM metadata from CZ_SEM tag."""
        if not self.is_sem:
            return None
        return self.pages.first.tags.valueof(34118)

    @property
    def sis_metadata(self) -> dict[str, Any] | None:
        """Olympus SIS metadata from OlympusSIS and OlympusINI tags."""
        if not self.pages.first.is_sis:
            return None
        tags = self.pages.first.tags
        result = {}
        try:
            result.update(tags.valueof(33471))  # OlympusINI
        except Exception:
            pass
        try:
            result.update(tags.valueof(33560))  # OlympusSIS
        except Exception:
            pass
        return result if result else None

    @cached_property
    def mdgel_metadata(self) -> dict[str, Any] | None:
        """MD-GEL metadata from MDFileTag tags."""
        if not self.is_mdgel:
            return None
        if 33445 in self.pages.first.tags:
            tags = self.pages.first.tags
        else:
            page = cast(TiffPage, self.pages[1])
            if 33445 in page.tags:
                tags = page.tags
            else:
                return None
        result = {}
        for code in range(33445, 33453):
            if code not in tags:
                continue
            name = TIFF.TAGS[code]
            result[name[2:]] = tags.valueof(code)
        return result

    @property
    def andor_metadata(self) -> dict[str, Any] | None:
        """Andor metadata from Andor tags."""
        return self.pages.first.andor_tags

    @property
    def epics_metadata(self) -> dict[str, Any] | None:
        """EPICS metadata from areaDetector tags."""
        return self.pages.first.epics_tags

    @property
    def tvips_metadata(self) -> dict[str, Any] | None:
        """TVIPS metadata from tag."""
        if not self.is_tvips:
            return None
        return self.pages.first.tags.valueof(37706)

    @cached_property
    def metaseries_metadata(self) -> dict[str, Any] | None:
        """MetaSeries metadata from ImageDescription tag of first tag."""
        # TODO: remove this? It is a per page property
        if not self.is_metaseries:
            return None
        return metaseries_description_metadata(self.pages.first.description)

    @cached_property
    def pilatus_metadata(self) -> dict[str, Any] | None:
        """Pilatus metadata from ImageDescription tag."""
        if not self.is_pilatus:
            return None
        return pilatus_description_metadata(self.pages.first.description)

    @cached_property
    def micromanager_metadata(self) -> dict[str, Any] | None:
        """Non-TIFF Micro-Manager metadata."""
        if not self.is_micromanager:
            return None
        return read_micromanager_metadata(self._fh)

    @cached_property
    def gdal_structural_metadata(self) -> dict[str, Any] | None:
        """Non-TIFF GDAL structural metadata."""
        return read_gdal_structural_metadata(self._fh)

    @cached_property
    def scanimage_metadata(self) -> dict[str, Any] | None:
        """ScanImage non-varying frame and ROI metadata.

        The returned dict may contain 'FrameData', 'RoiGroups', and 'version'
        keys.

        Varying frame data can be found in the ImageDescription tags.

        """
        if not self.is_scanimage:
            return None
        result: dict[str, Any] = {}
        try:
            framedata, roidata, version = read_scanimage_metadata(self._fh)
            result['version'] = version
            result['FrameData'] = framedata
            result.update(roidata)
        except ValueError:
            pass
        return result

    @property
    def geotiff_metadata(self) -> dict[str, Any] | None:
        """GeoTIFF metadata from tags."""
        if not self.is_geotiff:
            return None
        return self.pages.first.geotiff_tags

    @property
    def gdal_metadata(self) -> dict[str, Any] | None:
        """GDAL XML metadata from GDAL_METADATA tag."""
        if not self.is_gdal:
            return None
        return self.pages.first.tags.valueof(42112)

    @cached_property
    def astrotiff_metadata(self) -> dict[str, Any] | None:
        """AstroTIFF metadata from ImageDescription tag."""
        if not self.is_astrotiff:
            return None
        return astrotiff_description_metadata(self.pages.first.description)

    @cached_property
    def streak_metadata(self) -> dict[str, Any] | None:
        """Hamamatsu streak metadata from ImageDescription tag."""
        if not self.is_streak:
            return None
        return streak_description_metadata(
            self.pages.first.description, self.filehandle
        )

    @property
    def eer_metadata(self) -> str | None:
        """EER AcquisitionMetadata XML from tag 65001."""
        if not self.is_eer:
            return None
        value = self.pages.first.tags.valueof(65001)
        return None if value is None else value.decode()

