
class TiffPages(Sequence[TiffPage | TiffFrame]):
    """Sequence of TIFF image file directories (IFD chain).

    TiffPages instances have a state, such as a cache and keyframe, and are not
    thread-safe. All attributes are read-only.

    Parameters:
        arg:
            If a *TiffFile*, the file position must be at offset to offset to
            TiffPage.
            If a *TiffPage* or *TiffFrame*, page offsets are read from the
            SubIFDs tag.
            Only the first page is initially read from the file.
        index:
            Position of IFD chain in IFD tree.

    """

    parent: TiffFile | None = None
    """TiffFile instance pages belongs to."""

    _pages: list[TiffPage | TiffFrame | int]  # list of pages
    _keyframe: TiffPage | None
    _tiffpage: type[TiffPage] | type[TiffFrame]  # class used for reading pages
    _indexed: bool
    _cached: bool
    _cache: bool
    _offset: int
    _nextpageoffset: int | None
    _index: tuple[int, ...] | None

    def __init__(
        self,
        arg: TiffFile | TiffPage | TiffFrame,
        /,
        *,
        index: Sequence[int] | int | None = None,
    ) -> None:
        offset: int
        self.parent = None
        self._pages = []  # cache of TiffPages, TiffFrames, or their offsets
        self._indexed = False  # True if offsets to all pages were read
        self._cached = False  # True if all pages were read into cache
        self._tiffpage = TiffPage  # class used for reading pages
        self._keyframe = None  # page that is currently used as keyframe
        self._cache = False  # do not cache frames or pages (if not keyframe)
        self._offset = 0
        self._nextpageoffset = None

        if index is None:
            self._index = None
        elif isinstance(index, (int, numpy.integer)):
            self._index = (int(index),)
        else:
            self._index = tuple(index)

        if isinstance(arg, TiffFile):
            # read offset to first page from current file position
            self.parent = arg
            fh = self.parent.filehandle
            self._nextpageoffset = fh.tell()
            offset = struct.unpack(
                self.parent.tiff.offsetformat,
                fh.read(self.parent.tiff.offsetsize),
            )[0]
            if offset == 0:
                logger().warning(f'{arg!r} contains no pages')
                self._indexed = True
                return
        elif arg.subifds is not None:
            # use offsets from SubIFDs tag
            offsets = arg.subifds
            self.parent = arg.parent
            fh = self.parent.filehandle
            if len(offsets) == 0 or offsets[0] == 0:
                logger().warning(f'{arg!r} contains invalid SubIFDs')
                self._indexed = True
                return
            offset = offsets[0]
        else:
            self._indexed = True
            return

        self._offset = offset
        if offset >= fh.size:
            logger().warning(
                f'{self!r} invalid offset to first page {offset!r}'
            )
            self._indexed = True
            return

        pageindex: int | tuple[int, ...] = (
            0 if self._index is None else self._index + (0,)
        )

        # read and cache first page
        fh.seek(offset)
        page = TiffPage(self.parent, index=pageindex)
        self._pages.append(page)
        self._keyframe = page
        if self._nextpageoffset is None:
            # offsets from SubIFDs tag
            self._pages.extend(offsets[1:])
            self._indexed = True
            self._cached = True

    @property
    def pages(self) -> list[TiffPage | TiffFrame | int]:
        """Deprecated. Use the TiffPages sequence interface.

        :meta private:

        """
        warnings.warn(
            '<tifffile.TiffPages.pages> is deprecated since 2024.5.22. '
            'Use the TiffPages sequence interface.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self._pages

    @property
    def first(self) -> TiffPage:
        """First page as TiffPage if exists, else raise IndexError."""
        return cast(TiffPage, self._pages[0])

    @property
    def is_multipage(self) -> bool:
        """IFD chain contains more than one page."""
        try:
            self._seek(1)
            return True
        except IndexError:
            return False

    @property
    def cache(self) -> bool:
        """Pages and frames are being cached.

        When set to *False*, the cache is cleared.

        """
        return self._cache

    @cache.setter
    def cache(self, value: bool, /) -> None:
        value = bool(value)
        if self._cache and not value:
            self._clear()
        self._cache = value

    @property
    def useframes(self) -> bool:
        """Use TiffFrame (True) or TiffPage (False)."""
        return self._tiffpage == TiffFrame

    @useframes.setter
    def useframes(self, value: bool, /) -> None:
        self._tiffpage = TiffFrame if value else TiffPage

    @property
    def keyframe(self) -> TiffPage | None:
        """TiffPage used as keyframe for new TiffFrames."""
        return self._keyframe

    def set_keyframe(self, index: int, /) -> None:
        """Set keyframe to TiffPage specified by `index`.

        If not found in the cache, the TiffPage at `index` is loaded from file
        and added to the cache.

        """
        if not isinstance(index, (int, numpy.integer)):
            raise TypeError(f'indices must be integers, not {type(index)}')
        index = int(index)
        if index < 0:
            index %= len(self)
        if self._keyframe is not None and self._keyframe.index == index:
            return
        if index == 0:
            self._keyframe = cast(TiffPage, self._pages[0])
            return
        if self._indexed or index < len(self._pages):
            page = self._pages[index]
            if isinstance(page, TiffPage):
                self._keyframe = page
                return
            if isinstance(page, TiffFrame):
                # remove existing TiffFrame
                self._pages[index] = page.offset
        # load TiffPage from file
        tiffpage = self._tiffpage
        self._tiffpage = TiffPage
        try:
            self._keyframe = cast(TiffPage, self._getitem(index))
        finally:
            self._tiffpage = tiffpage
        # always cache keyframes
        self._pages[index] = self._keyframe

    @property
    def next_page_offset(self) -> int | None:
        """Offset where offset to new page can be stored."""
        if not self._indexed:
            self._seek(-1)
        return self._nextpageoffset

    def get(
        self,
        key: int,
        /,
        default: TiffPage | TiffFrame | None = None,
        *,
        validate: int = 0,
        cache: bool = False,
        aspage: bool = True,
    ) -> TiffPage | TiffFrame:
        """Return specified page from cache or file.

        The specified TiffPage or TiffFrame is read from file if it is not
        found in the cache.

        Parameters:
            key:
                Index of requested page in IFD chain.
            default:
                Page or frame to return if key is out of bounds.
                By default, an IndexError is raised if key is out of bounds.
            validate:
                If non-zero, raise RuntimeError if value does not match hash
                of TiffPage or TiffFrame.
            cache:
                Store returned page in cache for future use.
            aspage:
                Return TiffPage instance.

        """
        try:
            return self._getitem(
                key, validate=validate, cache=cache, aspage=aspage
            )
        except IndexError:
            if default is None:
                raise
        return default

    def _load(self, keyframe: TiffPage | bool | None = True, /) -> None:
        """Read all remaining pages from file."""
        assert self.parent is not None
        if self._cached:
            return
        pages = self._pages
        if not pages:
            return
        if not self._indexed:
            self._seek(-1)
        if not self._cache:
            return
        fh = self.parent.filehandle
        if keyframe is not None:
            keyframe = self._keyframe
        for i, page in enumerate(pages):
            if isinstance(page, (int, numpy.integer)):
                pageindex: int | tuple[int, ...] = (
                    i if self._index is None else self._index + (i,)
                )
                fh.seek(page)
                page = self._tiffpage(
                    self.parent, index=pageindex, keyframe=keyframe
                )
                pages[i] = page
        self._cached = True

    def _load_virtual_frames(self) -> None:
        """Calculate virtual TiffFrames."""
        assert self.parent is not None
        pages = self._pages
        try:
            if len(pages) > 1:
                raise ValueError('pages already loaded')
            page = cast(TiffPage, pages[0])
            if not page.is_contiguous:
                raise ValueError('data not contiguous')
            self._seek(4)
            # following pages are int
            delta = cast(int, pages[2]) - cast(int, pages[1])
            if (
                cast(int, pages[3]) - cast(int, pages[2]) != delta
                or cast(int, pages[4]) - cast(int, pages[3]) != delta
            ):
                raise ValueError('page offsets not equidistant')
            page1 = self._getitem(1, validate=page.hash)
            offsetoffset = page1.dataoffsets[0] - page1.offset
            if offsetoffset < 0 or offsetoffset > delta:
                raise ValueError('page offsets not equidistant')
            pages = [page, page1]
            filesize = self.parent.filehandle.size - delta

            for index, offset in enumerate(
                range(page1.offset + delta, filesize, delta)
            ):
                index += 2
                d = index * delta
                dataoffsets = tuple(i + d for i in page.dataoffsets)
                offset_or_none = offset if offset < 2**31 - 1 else None
                pages.append(
                    TiffFrame(
                        page.parent,
                        index=(
                            index
                            if self._index is None
                            else self._index + (index,)
                        ),
                        offset=offset_or_none,
                        dataoffsets=dataoffsets,
                        databytecounts=page.databytecounts,
                        keyframe=page,
                    )
                )
            self._pages = pages
            self._cache = True
            self._cached = True
            self._indexed = True
        except Exception as exc:
            if self.parent.filehandle.size >= 2147483648:
                logger().warning(
                    f'{self!r} <_load_virtual_frames> raised {exc!r:.128}'
                )

    def _clear(self, fully: bool = True, /) -> None:
        """Delete all but first page from cache. Set keyframe to first page."""
        pages = self._pages
        if not pages:
            return
        self._keyframe = cast(TiffPage, pages[0])
        if fully:
            # delete all but first TiffPage/TiffFrame
            for i, page in enumerate(pages[1:]):
                if not isinstance(page, int) and page.offset is not None:
                    pages[i + 1] = page.offset
        else:
            # delete only TiffFrames
            for i, page in enumerate(pages):
                if isinstance(page, TiffFrame) and page.offset is not None:
                    pages[i] = page.offset
        self._cached = False

    def _seek(self, index: int, /) -> int:
        """Seek file to offset of page specified by index and return offset."""
        assert self.parent is not None

        pages = self._pages
        lenpages = len(pages)
        if lenpages == 0:
            raise IndexError('index out of range')

        fh = self.parent.filehandle
        if fh.closed:
            raise ValueError('seek of closed file')

        if self._indexed or 0 <= index < lenpages:
            page = pages[index]
            offset = page if isinstance(page, int) else page.offset
            return fh.seek(offset)

        tiff = self.parent.tiff
        offsetformat = tiff.offsetformat
        offsetsize = tiff.offsetsize
        tagnoformat = tiff.tagnoformat
        tagnosize = tiff.tagnosize
        tagsize = tiff.tagsize
        unpack = struct.unpack

        page = pages[-1]
        offset = page if isinstance(page, int) else page.offset

        while lenpages < 2**32:
            # read offsets to pages from file until index is reached
            fh.seek(offset)
            # skip tags
            try:
                tagno = int(unpack(tagnoformat, fh.read(tagnosize))[0])
                if tagno > 4096:
                    raise TiffFileError(f'suspicious number of tags {tagno}')
            except Exception as exc:
                logger().error(
                    f'{self!r} corrupted tag list of page '
                    f'{lenpages} @{offset} raised {exc!r:.128}',
                )
                del pages[-1]
                lenpages -= 1
                self._indexed = True
                break
            self._nextpageoffset = offset + tagnosize + tagno * tagsize
            fh.seek(self._nextpageoffset)

            # read offset to next page
            try:
                offset = int(unpack(offsetformat, fh.read(offsetsize))[0])
            except Exception as exc:
                logger().error(
                    f'{self!r} invalid offset to page '
                    f'{lenpages + 1} @{self._nextpageoffset} '
                    f'raised {exc!r:.128}'
                )
                self._indexed = True
                break
            if offset == 0:
                self._indexed = True
                break
            if offset >= fh.size:
                logger().error(f'{self!r} invalid page offset {offset!r}')
                self._indexed = True
                break

            pages.append(offset)
            lenpages += 1
            if 0 <= index < lenpages:
                break

            # detect some circular references
            if lenpages == 100:
                for i, p in enumerate(pages[:-1]):
                    if offset == (p if isinstance(p, int) else p.offset):
                        index = i
                        self._pages = pages[: i + 1]
                        self._indexed = True
                        logger().error(
                            f'{self!r} invalid circular reference to IFD '
                            f'{i} at {offset=}'
                        )
                        break

        if index >= lenpages:
            raise IndexError('index out of range')

        page = pages[index]
        return fh.seek(page if isinstance(page, int) else page.offset)

    def _getlist(
        self,
        key: int | slice | Iterable[int] | None = None,
        /,
        useframes: bool = True,
        validate: bool = True,
    ) -> list[TiffPage | TiffFrame]:
        """Return specified pages as list of TiffPages or TiffFrames.

        The first item is a TiffPage, and is used as a keyframe for
        following TiffFrames.

        """
        getitem = self._getitem
        _useframes = self.useframes

        if key is None:
            key = iter(range(len(self)))
        elif isinstance(key, (int, numpy.integer)):
            # return single TiffPage
            key = int(key)
            self.useframes = False
            if key == 0:
                return [self.first]
            try:
                return [getitem(key)]
            finally:
                self.useframes = _useframes
        elif isinstance(key, slice):
            start, stop, _ = key.indices(2**31 - 1)
            if not self._indexed and max(stop, start) > len(self._pages):
                self._seek(-1)
            key = iter(range(*key.indices(len(self._pages))))
        elif isinstance(key, Iterable):
            key = iter(key)
        else:
            raise TypeError(
                f'key must be an integer, slice, or iterable, not {type(key)}'
            )

        # use first page as keyframe
        assert self._keyframe is not None
        keyframe = self._keyframe
        self.set_keyframe(next(key))
        validhash = self._keyframe.hash if validate else 0
        if useframes:
            self.useframes = True
        try:
            pages = [getitem(i, validate=validhash) for i in key]
            pages.insert(0, self._keyframe)
        finally:
            # restore state
            self._keyframe = keyframe
            if useframes:
                self.useframes = _useframes
        return pages

    def _getitem(
        self,
        key: int,
        /,
        *,
        validate: int = 0,  # hash
        cache: bool = False,
        aspage: bool = False,
    ) -> TiffPage | TiffFrame:
        """Return specified page from cache or file."""
        assert self.parent is not None
        key = int(key)
        pages = self._pages

        if key < 0:
            key %= len(self)
        elif self._indexed and key >= len(pages):
            raise IndexError(f'index {key} out of range({len(pages)})')

        tiffpage = TiffPage if aspage else self._tiffpage

        if key < len(pages):
            page = pages[key]
            if self._cache and not aspage:
                if not isinstance(page, (int, numpy.integer)):
                    if validate and validate != page.hash:
                        raise RuntimeError('page hash mismatch')
                    return page
            elif isinstance(page, (TiffPage, tiffpage)):
                # page is not an int
                if (
                    validate
                    and validate != page.hash  # type: ignore[union-attr]
                ):
                    raise RuntimeError('page hash mismatch')
                return page  # type: ignore[return-value]

        pageindex: int | tuple[int, ...] = (
            key if self._index is None else self._index + (key,)
        )
        self._seek(key)
        page = tiffpage(self.parent, index=pageindex, keyframe=self._keyframe)
        assert isinstance(page, (TiffPage, TiffFrame))
        if validate and validate != page.hash:
            raise RuntimeError('page hash mismatch')
        if self._cache or cache:
            pages[key] = page
        return page

    @overload
    def __getitem__(self, key: int, /) -> TiffPage | TiffFrame: ...

    @overload
    def __getitem__(
        self, key: slice | Iterable[int], /
    ) -> list[TiffPage | TiffFrame]: ...

    def __getitem__(
        self, key: int | slice | Iterable[int], /
    ) -> TiffPage | TiffFrame | list[TiffPage | TiffFrame]:
        pages = self._pages
        getitem = self._getitem

        if isinstance(key, (int, numpy.integer)):
            key = int(key)
            if key == 0:
                return cast(TiffPage, pages[key])
            return getitem(key)

        if isinstance(key, slice):
            start, stop, _ = key.indices(2**31 - 1)
            if not self._indexed and max(stop, start) > len(pages):
                self._seek(-1)
            return [getitem(i) for i in range(*key.indices(len(pages)))]

        if isinstance(key, Iterable):
            return [getitem(k) for k in key]

        raise TypeError('key must be an integer, slice, or iterable')

    def __iter__(self) -> Iterator[TiffPage | TiffFrame]:
        i = 0
        while True:
            try:
                yield self._getitem(i)
                i += 1
            except IndexError:
                break
        if self._cache:
            self._cached = True

    def __bool__(self) -> bool:
        """Return True if file contains any pages."""
        return len(self._pages) > 0

    def __len__(self) -> int:
        """Return number of pages in file."""
        if not self._indexed:
            self._seek(-1)
        return len(self._pages)

    def __repr__(self) -> str:
        return f'<tifffile.TiffPages @{self._offset}>'

@final
class TiffPageSeries(Sequence[TiffPage | TiffFrame | None]):
    """Sequence of TIFF pages making up multi-dimensional image.

    Many TIFF based formats, such as OME-TIFF, use series of TIFF pages to
    store chunks of larger, multi-dimensional images.
    The image shape and position of chunks in the multi-dimensional image is
    defined in format-specific metadata.
    All pages in a series must have the same :py:meth:`TiffPage.hash`,
    that is, the same shape, data type, and storage properties.
    Items of a series may be None (missing) or instances of
    :py:class:`TiffPage` or :py:class:`TiffFrame`, possibly belonging to
    different files.

    Parameters:
        pages:
            List of TiffPage, TiffFrame, or None.
            The file handles of TiffPages or TiffFrames may not be open.
        shape:
            Shape of image array in series.
        dtype:
            Data type of image array in series.
        axes:
            Character codes for dimensions in shape.
            Length must match shape.
        attr:
            Arbitrary metadata associated with series.
        index:
            Index of series in multi-series files.
        parent:
            TiffFile instance series belongs to.
        name:
            Name of series.
        kind:
            Nature of series, such as, 'ome' or 'imagej'.
        truncated:
            Series is truncated, for example, ImageJ hyperstack > 4 GB.
        multifile:
            Series contains pages from multiple files.
        squeeze:
            Remove length-1 dimensions (except X and Y) from shape and axes
            by default.
        transform:
            Function to transform image data after decoding.

    """

    levels: list[TiffPageSeries]
    """Multi-resolution, pyramidal levels. ``levels[0] is self``."""

    parent: TiffFile | None
    """TiffFile instance series belongs to."""

    keyframe: TiffPage
    """TiffPage of series."""

    dtype: numpy.dtype[Any]
    """Data type (native byte order) of image array in series."""

    kind: str
    """Nature of series."""

    name: str
    """Name of image series from metadata."""

    transform: Callable[[NDArray[Any]], NDArray[Any]] | None
    """Function to transform image data after decoding."""

    is_multifile: bool
    """Series contains pages from multiple files."""

    is_truncated: bool
    """Series contains single page describing multi-dimensional image."""

    _pages: list[TiffPage | TiffFrame | None]
    # List of pages in series.
    # Might contain only first page of contiguous series

    _index: int  # index of series in multi-series files
    _squeeze: bool
    _axes: str
    _axes_squeezed: str
    _shape: tuple[int, ...]
    _shape_squeezed: tuple[int, ...]
    _len: int
    _attr: dict[str, Any]

    def __init__(
        self,
        pages: Sequence[TiffPage | TiffFrame | None],
        /,
        shape: Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
        axes: str | None = None,
        *,
        attr: dict[str, Any] | None = None,
        coords: Mapping[str, NDArray[Any] | None] | None = None,
        index: int | None = None,
        parent: TiffFile | None = None,
        name: str | None = None,
        kind: str | None = None,
        truncated: bool = False,
        multifile: bool = False,
        squeeze: bool = True,
        transform: Callable[[NDArray[Any]], NDArray[Any]] | None = None,
    ) -> None:
        self._shape = ()
        self._shape_squeezed = ()
        self._axes = ''
        self._axes_squeezed = ''
        self._attr = {} if attr is None else dict(attr)

        self._index = int(index) if index else 0
        self._pages = list(pages)
        self.levels = [self]
        npages = len(self._pages)
        try:
            # find open TiffPage
            keyframe = next(
                p.keyframe
                for p in self._pages
                if p is not None
                and p.keyframe is not None
                and not p.keyframe.parent.filehandle.closed
            )
        except StopIteration:
            keyframe = next(
                p.keyframe
                for p in self._pages
                if p is not None and p.keyframe is not None
            )

        if shape is None:
            shape = keyframe.shape
        if axes is None:
            axes = keyframe.axes
        if dtype is None:
            dtype = keyframe.dtype

        self.dtype = numpy.dtype(dtype)
        self.kind = kind if kind else ''
        self.name = name if name else ''
        self.transform = transform
        self.keyframe = keyframe
        self.is_multifile = bool(multifile)
        self.is_truncated = bool(truncated)

        if parent is not None:
            self.parent = parent
        elif self._pages:
            self.parent = self.keyframe.parent
        else:
            self.parent = None

        self._set_dimensions(shape, axes, coords, squeeze)

        if not truncated and npages == 1:
            s = product(keyframe.shape)
            if s > 0:
                self._len = int(product(self.shape) // s)
            else:
                self._len = npages
        else:
            self._len = npages

    def _set_dimensions(
        self,
        shape: Sequence[int],
        axes: str,
        coords: Mapping[str, NDArray[Any] | None] | None = None,
        squeeze: bool = True,
        /,
    ) -> None:
        """Set shape, axes, and coords."""
        self._squeeze = bool(squeeze)
        self._shape = tuple(shape)
        self._axes = axes
        self._shape_squeezed, self._axes_squeezed, _ = squeeze_axes(
            shape, axes
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of image array in series."""
        return self._shape_squeezed if self._squeeze else self._shape

    @property
    def axes(self) -> str:
        """Character codes for dimensions in image array."""
        return self._axes_squeezed if self._squeeze else self._axes

    @property
    def coords(self) -> dict[str, NDArray[Any]]:
        """Ordered map of dimension names to coordinate arrays."""
        raise NotImplementedError
        # return {
        #     name: numpy.arange(size)
        #     for name, size in zip(self.dims, self.shape)
        # }

    def get_shape(self, squeeze: bool | None = None) -> tuple[int, ...]:
        """Return default, squeezed, or expanded shape of series.

        Parameters:
            squeeze: Remove length-1 dimensions from shape.

        """
        if squeeze is None:
            squeeze = self._squeeze
        return self._shape_squeezed if squeeze else self._shape

    def get_axes(self, squeeze: bool | None = None) -> str:
        """Return default, squeezed, or expanded axes of series.

        Parameters:
            squeeze: Remove length-1 dimensions from axes.

        """
        if squeeze is None:
            squeeze = self._squeeze
        return self._axes_squeezed if squeeze else self._axes

    def get_coords(
        self, squeeze: bool | None = None
    ) -> dict[str, NDArray[Any]]:
        """Return default, squeezed, or expanded coords of series.

        Parameters:
            squeeze: Remove length-1 dimensions from coords.

        """
        raise NotImplementedError

    def asarray(
        self, *, level: int | None = None, **kwargs: Any
    ) -> NDArray[Any]:
        """Return images from series of pages as NumPy array.

        Parameters:
            level:
                Pyramid level to return.
                By default, the base layer is returned.
            **kwargs:
                Additional arguments passed to :py:meth:`TiffFile.asarray`.

        """
        if self.parent is None:
            raise ValueError('no parent')
        if level is not None:
            return self.levels[level].asarray(**kwargs)
        result = self.parent.asarray(series=self, **kwargs)
        if self.transform is not None:
            result = self.transform(result)
        return result

    def aszarr(
        self, *, level: int | None = None, **kwargs: Any
    ) -> ZarrTiffStore:
        """Return image array from series of pages as Zarr 2 store.

        Parameters:
            level:
                Pyramid level to return.
                By default, a multi-resolution store is returned.
            **kwargs:
                Additional arguments passed to :py:class:`ZarrTiffStore`.

        """
        if self.parent is None:
            raise ValueError('no parent')
        return ZarrTiffStore(self, level=level, **kwargs)

    @cached_property
    def dataoffset(self) -> int | None:
        """Offset to contiguous image data in file."""
        if not self._pages:
            return None
        pos = 0
        for page in self._pages:
            if page is None or len(page.dataoffsets) == 0:
                return None
            if not page.is_final:
                return None
            if not pos:
                pos = page.dataoffsets[0] + page.nbytes
                continue
            if pos != page.dataoffsets[0]:
                return None
            pos += page.nbytes

        page = self._pages[0]
        if page is None or len(page.dataoffsets) == 0:
            return None
        offset = page.dataoffsets[0]
        if (
            len(self._pages) == 1
            and isinstance(page, TiffPage)
            and (page.is_imagej or page.is_shaped or page.is_stk)
        ):
            # truncated files
            return offset
        if pos == offset + product(self.shape) * self.dtype.itemsize:
            return offset
        return None

    @property
    def is_pyramidal(self) -> bool:
        """Series contains multiple resolutions."""
        return len(self.levels) > 1

    @cached_property
    def attr(self) -> dict[str, Any]:
        """Arbitrary metadata associated with series."""
        return self._attr

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return len(self.shape)

    @property
    def dims(self) -> tuple[str, ...]:
        """Names of dimensions in image array."""
        # return tuple(self.coords.keys())
        return tuple(
            unique_strings(TIFF.AXES_NAMES.get(ax, ax) for ax in self.axes)
        )

    @property
    def sizes(self) -> dict[str, int]:
        """Ordered map of dimension names to lengths."""
        # return dict(zip(self.coords.keys(), self.shape))
        return dict(zip(self.dims, self.shape))

    @cached_property
    def size(self) -> int:
        """Number of elements in array."""
        return product(self.shape)

    @cached_property
    def nbytes(self) -> int:
        """Number of bytes in array."""
        return self.size * self.dtype.itemsize

    @property
    def pages(self) -> TiffPageSeries:
        # sequence of TiffPages or TiffFrame in series
        # a workaround to keep the old interface working
        return self

    def _getitem(self, key: int, /) -> TiffPage | TiffFrame | None:
        """Return specified page of series from cache or file."""
        key = int(key)
        if key < 0:
            key %= self._len
        if len(self._pages) == 1 and 0 < key < self._len:
            page = self._pages[0]
            assert page is not None
            assert self.parent is not None
            return self.parent.pages._getitem(page.index + key)
        return self._pages[key]

    @overload
    def __getitem__(
        self, key: int | numpy.integer[Any], /
    ) -> TiffPage | TiffFrame | None: ...

    @overload
    def __getitem__(
        self, key: slice | Iterable[int], /
    ) -> list[TiffPage | TiffFrame | None]: ...

    def __getitem__(
        self, key: int | numpy.integer[Any] | slice | Iterable[int], /
    ) -> TiffPage | TiffFrame | list[TiffPage | TiffFrame | None] | None:
        """Return specified page(s)."""
        if isinstance(key, (int, numpy.integer)):
            return self._getitem(int(key))
        if isinstance(key, slice):
            return [self._getitem(i) for i in range(*key.indices(self._len))]
        if isinstance(key, Iterable) and not isinstance(key, str):
            return [self._getitem(k) for k in key]
        raise TypeError('key must be an integer, slice, or iterable')

    def __iter__(self) -> Iterator[TiffPage | TiffFrame | None]:
        """Return iterator over pages in series."""
        if len(self._pages) == self._len:
            yield from self._pages
        else:
            assert self.parent is not None and self._pages[0] is not None
            pages = self.parent.pages
            index = self._pages[0].index
            for i in range(self._len):
                yield pages[index + i]

    def __len__(self) -> int:
        """Return number of pages in series."""
        return self._len

    def __repr__(self) -> str:
        return f'<tifffile.TiffPageSeries {self._index} {self.kind}>'

    def __str__(self) -> str:
        s = '  '.join(
            s
            for s in (
                snipstr(f'{self.name!r}', 20) if self.name else '',
                'x'.join(str(i) for i in self.shape),
                str(self.dtype),
                self.axes,
                self.kind,
                (f'{len(self.levels)} Levels') if self.is_pyramidal else '',
                f'{len(self)} Pages',
                (f'@{self.dataoffset}') if self.dataoffset else '',
            )
            if s
        )
        return f'TiffPageSeries {self._index}  {s}'
