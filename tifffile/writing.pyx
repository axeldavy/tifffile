
class TiffFileError(Exception):
    """Exception to indicate invalid TIFF structure."""


@final
class TiffWriter:
    """Write NumPy arrays to TIFF file.

    TiffWriter's main purpose is saving multi-dimensional NumPy arrays in
    TIFF containers, not to create any possible TIFF format.
    Specifically, ExifIFD and GPSIFD tags are not supported.

    TiffWriter instances must be closed with :py:meth:`TiffWriter.close`,
    which is automatically called when using the 'with' context manager.

    TiffWriter instances are not thread-safe. All attributes are read-only.

    Parameters:
        file:
            Specifies file to write.
        mode:
            Binary file open mode if `file` is file name.
            The default is 'w', which opens files for writing, truncating
            existing files.
            'x' opens files for exclusive creation, failing on existing files.
            'r+' opens files for updating, enabling `append`.
        bigtiff:
            Write 64-bit BigTIFF formatted file, which can exceed 4 GB.
            By default, a classic 32-bit TIFF file is written, which is
            limited to 4 GB.
            If `append` is *True*, the existing file's format is used.
        byteorder:
            Endianness of TIFF format. One of '<', '>', '=', or '|'.
            The default is the system's native byte order.
        append:
            If `file` is existing standard TIFF file, append image data
            and tags to file.
            Parameters `bigtiff` and `byteorder` set from existing file.
            Appending does not scale well with the number of pages already in
            the file and may corrupt specifically formatted TIFF files such as
            OME-TIFF, LSM, STK, ImageJ, or FluoView.
        imagej:
            Write ImageJ hyperstack compatible file if `ome` is not enabled.
            This format can handle data types uint8, uint16, or float32 and
            data shapes up to 6 dimensions in TZCYXS order.
            RGB images (S=3 or S=4) must be `uint8`.
            ImageJ's default byte order is big-endian, but this
            implementation uses the system's native byte order by default.
            ImageJ hyperstacks do not support BigTIFF or compression.
            The ImageJ file format is undocumented.
            Use FIJI's Bio-Formats import function for compressed files.
        ome:
            Write OME-TIFF compatible file.
            By default, the OME-TIFF format is used if the file name extension
            contains '.ome.', `imagej` is not enabled, and the `description`
            argument in the first call of :py:meth:`TiffWriter.write` is not
            specified.
            The format supports multiple, up to 9 dimensional image series.
            The default axes order is TZC(S)YX(S).
            Refer to the OME model for restrictions of this format.
        shaped:
            Write tifffile "shaped" compatible file.
            The shape of multi-dimensional images is stored in JSON format in
            a ImageDescription tag of the first page of a series.
            This is the default format used by tifffile unless `imagej` or
            `ome` are enabled or ``metadata=None`` is passed to
            :py:meth:`TiffWriter.write`.

    Raises:
        ValueError:
            The TIFF file cannot be appended to. Use ``append='force'`` to
            force appending, which may result in a corrupted file.

    """

    tiff: TiffFormat
    """Format of TIFF file being written."""

    _fh: FileHandle
    _omexml: OmeXml | None
    _ome: bool | None  # writing OME-TIFF format
    _imagej: bool  # writing ImageJ format
    _tifffile: bool  # writing Tifffile shaped format
    _truncate: bool
    _metadata: dict[str, Any] | None
    _colormap: NDArray[numpy.uint16] | None
    _tags: list[tuple[int, bytes, Any, bool]] | None
    _datashape: tuple[int, ...] | None  # shape of data in consecutive pages
    _datadtype: numpy.dtype[Any] | None  # data type
    _dataoffset: int | None  # offset to data
    _databytecounts: list[int] | None  # byte counts per plane
    _dataoffsetstag: int | None  # strip or tile offset tag code
    _descriptiontag: TiffTag | None  # TiffTag for updating comment
    _ifdoffset: int
    _subifds: int  # number of subifds
    _subifdslevel: int  # index of current subifd level
    _subifdsoffsets: list[int]  # offsets to offsets to subifds
    _nextifdoffsets: list[int]  # offsets to offset to next ifd
    _ifdindex: int  # index of current ifd
    _storedshape: StoredShape | None  # normalized shape in consecutive pages

    def __init__(
        self,
        file: str | os.PathLike[Any] | FileHandle | IO[bytes],
        /,
        *,
        mode: Literal['w', 'x', 'r+'] | None = None,
        bigtiff: bool = False,
        byteorder: ByteOrder | None = None,
        append: bool | str = False,
        imagej: bool = False,
        ome: bool | None = None,
        shaped: bool | None = None,
    ) -> None:
        if mode in {'r+', 'r+b'} or (
            isinstance(file, FileHandle) and file._mode == 'r+b'
        ):
            mode = 'r+'
            append = True
        if append:
            # determine if file is an existing TIFF file that can be extended
            try:
                with FileHandle(file, mode='rb', size=0) as fh:
                    pos = fh.tell()
                    try:
                        with TiffFile(fh) as tif:
                            if append != 'force' and not tif.is_appendable:
                                raise ValueError(
                                    'cannot append to file containing metadata'
                                )
                            byteorder = tif.byteorder
                            bigtiff = tif.is_bigtiff
                            self._ifdoffset = cast(
                                int, tif.pages.next_page_offset
                            )
                    finally:
                        fh.seek(pos)
                    append = True
            except (OSError, FileNotFoundError):
                append = False

        if append:
            if mode not in {None, 'r+', 'r+b'}:
                raise ValueError("append mode must be 'r+'")
            mode = 'r+'
        elif mode is None:
            mode = 'w'

        if byteorder is None or byteorder in {'=', '|'}:
            byteorder = '<' if sys.byteorder == 'little' else '>'
        elif byteorder not in {'<', '>'}:
            raise ValueError(f'invalid byteorder {byteorder}')

        if byteorder == '<':
            self.tiff = TIFF.BIG_LE if bigtiff else TIFF.CLASSIC_LE
        else:
            self.tiff = TIFF.BIG_BE if bigtiff else TIFF.CLASSIC_BE

        self._truncate = False
        self._metadata = None
        self._colormap = None
        self._tags = None
        self._datashape = None
        self._datadtype = None
        self._dataoffset = None
        self._databytecounts = None
        self._dataoffsetstag = None
        self._descriptiontag = None
        self._subifds = 0
        self._subifdslevel = -1
        self._subifdsoffsets = []
        self._nextifdoffsets = []
        self._ifdindex = 0
        self._omexml = None
        self._storedshape = None

        self._fh = FileHandle(file, mode=mode, size=0)
        if append:
            self._fh.seek(0, os.SEEK_END)
        else:
            assert byteorder is not None
            self._fh.write(b'II' if byteorder == '<' else b'MM')
            if bigtiff:
                self._fh.write(struct.pack(byteorder + 'HHH', 43, 8, 0))
            else:
                self._fh.write(struct.pack(byteorder + 'H', 42))
            # first IFD
            self._ifdoffset = self._fh.tell()
            self._fh.write(struct.pack(self.tiff.offsetformat, 0))

        self._ome = None if ome is None else bool(ome)
        self._imagej = False if self._ome else bool(imagej)
        if self._imagej:
            self._ome = False
        if self._ome or self._imagej:
            self._tifffile = False
        else:
            self._tifffile = True if shaped is None else bool(shaped)

        if imagej and bigtiff:
            warnings.warn(
                f'{self!r} writing nonconformant BigTIFF ImageJ', UserWarning
            )

    def write(
        self,
        data: (
            ArrayLike | Iterator[NDArray[Any] | None] | Iterator[bytes] | None
        ) = None,
        *,
        shape: Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
        photometric: PHOTOMETRIC | int | str | None = None,
        planarconfig: PLANARCONFIG | int | str | None = None,
        extrasamples: Sequence[EXTRASAMPLE | int | str] | None = None,
        volumetric: bool = False,
        tile: Sequence[int] | None = None,
        rowsperstrip: int | None = None,
        bitspersample: int | None = None,
        compression: COMPRESSION | int | str | bool | None = None,
        compressionargs: dict[str, Any] | None = None,
        predictor: PREDICTOR | int | str | bool | None = None,
        subsampling: tuple[int, int] | None = None,
        jpegtables: bytes | None = None,
        iccprofile: bytes | None = None,
        colormap: ArrayLike | None = None,
        description: str | bytes | None = None,
        datetime: str | bool | DateTime | None = None,
        resolution: (
            tuple[float | tuple[int, int], float | tuple[int, int]] | None
        ) = None,
        resolutionunit: RESUNIT | int | str | None = None,
        subfiletype: FILETYPE | int | None = None,
        software: str | bytes | bool | None = None,
        subifds: int | Sequence[int] | None = None,
        metadata: dict[str, Any] | None = {},
        extratags: Sequence[TagTuple] | None = None,
        contiguous: bool = False,
        truncate: bool = False,
        align: int | None = None,
        maxworkers: int | None = None,
        buffersize: int | None = None,
        returnoffset: bool = False,
    ) -> tuple[int, int] | None:
        r"""Write multi-dimensional image to series of TIFF pages.

        Metadata in JSON, ImageJ, or OME-XML format are written to the
        ImageDescription tag of the first page of a series by default,
        such that the image can later be read back as an array of the
        same shape.

        The values of the ImageWidth, ImageLength, ImageDepth, and
        SamplesPerPixel tags are inferred from the last dimensions of the
        data's shape.
        The value of the SampleFormat tag is inferred from the data's dtype.
        Image data are written uncompressed in one strip per plane by default.
        Dimensions higher than 2 to 4 (depending on photometric mode, planar
        configuration, and volumetric mode) are flattened and written as
        separate pages.
        If the data size is zero, write a single page with shape (0, 0).

        Parameters:
            data:
                Specifies image to write.
                If *None*, an empty image is written, which size and type must
                be specified using `shape` and `dtype` arguments.
                This option cannot be used with compression, predictors,
                packed integers, or bilevel images.
                A copy of array-like data is made if it is not a C-contiguous
                numpy or dask array with the same byteorder as the TIFF file.
                Iterators must yield ndarrays or bytes compatible with the
                file's byteorder as well as the `shape` and `dtype` arguments.
                Iterator bytes must be compatible with the `compression`,
                `predictor`, `subsampling`, and `jpegtables` arguments.
                If `tile` is specified, iterator items must match the tile
                shape. Incomplete tiles are zero-padded.
                Iterators of non-tiled images must yield ndarrays of
                `shape[1:]` or strips as bytes. Iterators of strip ndarrays
                are not supported.
                Writing dask arrays might be excruciatingly slow for arrays
                with many chunks or files with many segments.
                (https://github.com/dask/dask/issues/8570).
            shape:
                Shape of image to write.
                The default is inferred from the `data` argument if possible.
                A ValueError is raised if the value is incompatible with
                the `data` or other arguments.
            dtype:
                NumPy data type of image to write.
                The default is inferred from the `data` argument if possible.
                A ValueError is raised if the value is incompatible with
                the `data` argument.
            photometric:
                Color space of image.
                The default is inferred from the data shape, dtype, and the
                `colormap` argument.
                A UserWarning is logged if RGB color space is auto-detected.
                Specify this parameter to silence the warning and to avoid
                ambiguities.
                *MINISBLACK*: for bilevel and grayscale images, 0 is black.
                *MINISWHITE*: for bilevel and grayscale images, 0 is white.
                *RGB*: the image contains red, green and blue samples.
                *SEPARATED*: the image contains CMYK samples.
                *PALETTE*: the image is used as an index into a colormap.
                *CFA*: the image is a Color Filter Array. The
                CFARepeatPatternDim, CFAPattern, and other DNG or TIFF/EP tags
                must be specified in `extratags` to produce a valid file.
                The value is written to the PhotometricInterpretation tag.
            planarconfig:
                Specifies if samples are stored interleaved or in separate
                planes.
                *CONTIG*: the last dimension contains samples.
                *SEPARATE*: the 3rd or 4th last dimension contains samples.
                The default is inferred from the data shape and `photometric`
                mode.
                If this parameter is set, extra samples are used to store
                grayscale images.
                The value is written to the PlanarConfiguration tag.
            extrasamples:
                Interpretation of extra components in pixels.
                *UNSPECIFIED*: no transparency information (default).
                *ASSOCALPHA*: true transparency with premultiplied color.
                *UNASSALPHA*: independent transparency masks.
                The values are written to the ExtraSamples tag.
            volumetric:
                Write volumetric image to single page (instead of multiple
                pages) using SGI ImageDepth tag.
                The volumetric format is not part of the TIFF specification,
                and few software can read it.
                OME and ImageJ formats are not compatible with volumetric
                storage.
            tile:
                Shape ([depth,] length, width) of image tiles to write.
                By default, image data are written in strips.
                The tile length and width must be a multiple of 16.
                If a tile depth is provided, the SGI ImageDepth and TileDepth
                tags are used to write volumetric data.
                Tiles cannot be used to write contiguous series, except if
                the tile shape matches the data shape.
                The values are written to the TileWidth, TileLength, and
                TileDepth tags.
            rowsperstrip:
                Number of rows per strip.
                By default, strips are about 256 KB if `compression` is
                enabled, else rowsperstrip is set to the image length.
                The value is written to the RowsPerStrip tag.
            bitspersample:
                Number of bits per sample.
                The default is the number of bits of the data's dtype.
                Different values per samples are not supported.
                Unsigned integer data are packed into bytes as tightly as
                possible.
                Valid values are 1-8 for uint8, 9-16 for uint16, and 17-32
                for uint32.
                This setting cannot be used with compression, contiguous
                series, or empty files.
                The value is written to the BitsPerSample tag.
            compression:
                Compression scheme used on image data.
                By default, image data are written uncompressed.
                Compression cannot be used to write contiguous series.
                Compressors may require certain data shapes, types or value
                ranges. For example, JPEG compression requires grayscale or
                RGB(A), uint8 or 12-bit uint16.
                JPEG compression is experimental. JPEG markers and TIFF tags
                may not match.
                Only a limited set of compression schemes are implemented.
                'ZLIB' is short for ADOBE_DEFLATE.
                The value is written to the Compression tag.
            compressionargs:
                Extra arguments passed to compression codec, for example,
                compression level. Refer to the Imagecodecs implementation
                for supported arguments.
            predictor:
                Horizontal differencing operator applied to image data before
                compression.
                By default, no operator is applied.
                Predictors can only be used with certain compression schemes
                and data types.
                The value is written to the Predictor tag.
            subsampling:
                Horizontal and vertical subsampling factors used for the
                chrominance components of images: (1, 1), (2, 1), (2, 2), or
                (4, 1). The default is *(2, 2)*.
                Currently applies to JPEG compression of RGB images only.
                Images are stored in YCbCr color space, the value of the
                PhotometricInterpretation tag is *YCBCR*.
                Segment widths must be a multiple of 8 times the horizontal
                factor. Segment lengths and rowsperstrip must be a multiple
                of 8 times the vertical factor.
                The values are written to the YCbCrSubSampling tag.
            jpegtables:
                JPEG quantization and/or Huffman tables.
                Use for copying pre-compressed JPEG segments.
                The value is written to the JPEGTables tag.
            iccprofile:
                International Color Consortium (ICC) device profile
                characterizing image color space.
                The value is written verbatim to the InterColorProfile tag.
            colormap:
                RGB color values for corresponding data value.
                The colormap array must be of shape
                `(3, 2\*\*(data.itemsize*8))` (or `(3, 256)` for ImageJ)
                and dtype uint16.
                The image's data type must be uint8 or uint16 (or float32
                for ImageJ) and the values are indices into the last
                dimension of the colormap.
                The value is written to the ColorMap tag.
            description:
                Subject of image. Must be 7-bit ASCII.
                Cannot be used with the ImageJ or OME formats.
                The value is written to the ImageDescription tag of the
                first page of a series.
            datetime:
                Date and time of image creation in ``%Y:%m:%d %H:%M:%S``
                format or datetime object.
                If *True*, the current date and time is used.
                The value is written to the DateTime tag of the first page
                of a series.
            resolution:
                Number of pixels per `resolutionunit` in X and Y directions
                as float or rational numbers.
                The default is (1.0, 1.0).
                The values are written to the YResolution and XResolution tags.
            resolutionunit:
                Unit of measurement for `resolution` values.
                The default is *NONE* if `resolution` is not specified and
                for ImageJ format, else *INCH*.
                The value is written to the ResolutionUnit tags.
            subfiletype:
                Bitfield to indicate kind of image.
                Set bit 0 if the image is a reduced-resolution version of
                another image.
                Set bit 1 if the image is part of a multi-page image.
                Set bit 2 if the image is transparency mask for another
                image (photometric must be MASK, SamplesPerPixel and
                bitspersample must be 1).
            software:
                Name of software used to create file.
                Must be 7-bit ASCII. The default is 'tifffile.py'.
                Unless *False*, the value is written to the Software tag of
                the first page of a series.
            subifds:
                Number of child IFDs.
                If greater than 0, the following `subifds` number of series
                are written as child IFDs of the current series.
                The number of IFDs written for each SubIFD level must match
                the number of IFDs written for the current series.
                All pages written to a certain SubIFD level of the current
                series must have the same hash.
                SubIFDs cannot be used with truncated or ImageJ files.
                SubIFDs in OME-TIFF files must be sub-resolutions of the
                main IFDs.
            metadata:
                Additional metadata describing image, written along
                with shape information in JSON, OME-XML, or ImageJ formats
                in ImageDescription or IJMetadata tags.
                If *None*, or the `shaped` argument to :py:class:`TiffWriter`
                is *False*, no information in JSON format is written to
                the ImageDescription tag.
                The 'axes' item defines the character codes for dimensions in
                `data` or `shape`.
                Refer to :py:class:`OmeXml` for supported keys when writing
                OME-TIFF.
                Refer to :py:func:`imagej_description` and
                :py:func:`imagej_metadata_tag` for items supported
                by the ImageJ format. Items 'Info', 'Labels', 'Ranges',
                'LUTs', 'Plot', 'ROI', and 'Overlays' are written to the
                IJMetadata and IJMetadataByteCounts tags.
                Strings must be 7-bit ASCII.
                Written with the first page of a series only.
            extratags:
                Additional tags to write. A list of tuples with 5 items:

                0. code (int): Tag Id.
                1. dtype (:py:class:`DATATYPE`):
                   Data type of items in `value`.
                2. count (int): Number of data values.
                   Not used for string or bytes values.
                3. value (Sequence[Any]): `count` values compatible with
                   `dtype`. Bytes must contain count values of dtype packed
                   as binary data.
                4. writeonce (bool): If *True*, write tag to first page
                   of a series only.

                Duplicate and select tags in TIFF.TAG_FILTERED are not written
                if the extratag is specified by integer code.
                Extratags cannot be used to write IFD type tags.

            contiguous:
                If *False* (default), write data to a new series.
                If *True* and the data and arguments are compatible with
                previous written ones (same shape, no compression, etc.),
                the image data are stored contiguously after the previous one.
                In that case, `photometric`, `planarconfig`, and
                `rowsperstrip` are ignored.
                Metadata such as `description`, `metadata`, `datetime`,
                and `extratags` are written to the first page of a contiguous
                series only.
                Contiguous mode cannot be used with the OME or ImageJ formats.
            truncate:
                If *True*, only write first page of contiguous series
                if possible (uncompressed, contiguous, not tiled).
                Other TIFF readers will only be able to read part of the data.
                Cannot be used with the OME or ImageJ formats.
            align:
                Byte boundary on which to align image data in file.
                The default is 16.
                Use mmap.ALLOCATIONGRANULARITY for memory-mapped data.
                Following contiguous writes are not aligned.
            maxworkers:
                Maximum number of threads to concurrently compress tiles
                or strips.
                If *None* or *0*, use up to :py:attr:`_TIFF.MAXWORKERS` CPU
                cores for compressing large segments.
                Using multiple threads can significantly speed up this
                function if the bottleneck is encoding the data, for example,
                in case of large JPEG compressed tiles.
                If the bottleneck is I/O or pure Python code, using multiple
                threads might be detrimental.
            buffersize:
                Approximate number of bytes to compress in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE` * 2.
            returnoffset:
                Return offset and number of bytes of memory-mappable image
                data in file.

        Returns:
            If `returnoffset` is *True* and the image data in the file are
            memory-mappable, return the offset and number of bytes of the
            image data in the file.

        """
        # TODO: refactor this function

        fh: FileHandle
        storedshape: StoredShape = StoredShape(frames=-1)
        byteorder: Literal['>', '<']
        inputshape: tuple[int, ...]
        datashape: tuple[int, ...]
        dataarray: NDArray[Any] | None = None
        dataiter: Iterator[NDArray[Any] | bytes | None] | None = None
        dataoffsetsoffset: tuple[int, int | None] | None = None
        databytecountsoffset: tuple[int, int | None] | None = None
        subifdsoffsets: tuple[int, int | None] | None = None
        datadtype: numpy.dtype[Any]
        bilevel: bool
        tiles: tuple[int, ...]
        ifdpos: int
        photometricsamples: int
        pos: int | None = None
        predictortag: int
        predictorfunc: Callable[..., Any] | None = None
        compressiontag: int
        compressionfunc: Callable[..., Any] | None = None
        tags: list[tuple[int, bytes, bytes | None, bool]]
        numtiles: int
        numstrips: int

        fh = self._fh
        byteorder = self.tiff.byteorder

        if data is None:
            # empty
            if shape is None or dtype is None:
                raise ValueError(
                    "missing required 'shape' or 'dtype' arguments"
                )
            dataarray = None
            dataiter = None
            datashape = tuple(shape)
            datadtype = numpy.dtype(dtype).newbyteorder(byteorder)

        elif hasattr(data, '__next__'):
            # iterator/generator
            if shape is None or dtype is None:
                raise ValueError(
                    "missing required 'shape' or 'dtype' arguments"
                )
            dataiter = data  # type: ignore[assignment]
            datashape = tuple(shape)
            datadtype = numpy.dtype(dtype).newbyteorder(byteorder)

        elif hasattr(data, 'dtype'):
            # numpy, zarr, or dask array
            data = cast(numpy.ndarray, data)  # type: ignore[type-arg]
            dataarray = data
            datadtype = numpy.dtype(data.dtype).newbyteorder(byteorder)
            if not hasattr(data, 'reshape'):
                # zarr array cannot be shape-normalized
                dataarray = numpy.asarray(data, datadtype, 'C')
            else:
                try:
                    # numpy array must be C contiguous
                    if data.flags.f_contiguous:
                        dataarray = numpy.asarray(data, datadtype, 'C')
                except AttributeError:
                    # not a numpy array
                    pass
            datashape = dataarray.shape
            dataiter = None
            if dtype is not None and numpy.dtype(dtype) != datadtype:
                raise ValueError(
                    f'dtype argument {dtype!r} does not match '
                    f'data dtype {datadtype}'
                )
            if shape is not None and shape != dataarray.shape:
                raise ValueError(
                    f'shape argument {shape!r} does not match '
                    f'data shape {dataarray.shape}'
                )

        else:
            # scalar, list, tuple, etc
            # if dtype is not specified, default to float64
            datadtype = numpy.dtype(dtype).newbyteorder(byteorder)
            dataarray = numpy.asarray(data, datadtype, 'C')
            datashape = dataarray.shape
            dataiter = None

        del data

        if any(size >= 4294967296 for size in datashape):
            raise ValueError('invalid data shape')

        bilevel = datadtype.char == '?'
        if bilevel:
            index = -1 if datashape[-1] > 1 else -2
            datasize = product(datashape[:index])
            if datashape[index] % 8:
                datasize *= datashape[index] // 8 + 1
            else:
                datasize *= datashape[index] // 8
        else:
            datasize = product(datashape) * datadtype.itemsize

        if datasize == 0:
            dataarray = None
            compression = False
            bitspersample = None
            if metadata is not None:
                truncate = True

        if (
            not compression
            or (
                not isinstance(compression, bool)  # because True == 1
                and compression in ('NONE', 'None', 'none', 1)
            )
            or (
                isinstance(compression, (tuple, list))
                and compression[0] in (None, 0, 1, 'NONE', 'None', 'none')
            )
        ):
            compression = False

        if not predictor or (
            not isinstance(predictor, bool)  # because True == 1
            and predictor in {'NONE', 'None', 'none', 1}
        ):
            predictor = False

        inputshape = datashape

        packints = (
            bitspersample is not None
            and bitspersample != datadtype.itemsize * 8
        )

        # just append contiguous data if possible
        if self._datashape is not None and self._datadtype is not None:
            if colormap is not None:
                colormap = numpy.asarray(colormap, dtype=byteorder + 'H')
            if (
                not contiguous
                or self._datashape[1:] != datashape
                or self._datadtype != datadtype
                or (colormap is None and self._colormap is not None)
                or (self._colormap is None and colormap is not None)
                or not numpy.array_equal(
                    colormap, self._colormap  # type: ignore[arg-type]
                )
            ):
                # incompatible shape, dtype, or colormap
                self._write_remaining_pages()

                if self._imagej:
                    raise ValueError(
                        'the ImageJ format does not support '
                        'non-contiguous series'
                    )
                if self._omexml is not None:
                    if self._subifdslevel < 0:
                        # add image to OME-XML
                        assert self._storedshape is not None
                        assert self._metadata is not None
                        self._omexml.addimage(
                            dtype=self._datadtype,
                            shape=self._datashape[
                                0 if self._datashape[0] != 1 else 1 :
                            ],
                            storedshape=self._storedshape.shape,
                            **self._metadata,
                        )
                elif metadata is not None:
                    self._write_image_description()
                    # description might have been appended to file
                    fh.seek(0, os.SEEK_END)

                if self._subifds:
                    if self._truncate or truncate:
                        raise ValueError(
                            'SubIFDs cannot be used with truncated series'
                        )
                    self._subifdslevel += 1
                    if self._subifdslevel == self._subifds:
                        # done with writing SubIFDs
                        self._nextifdoffsets = []
                        self._subifdsoffsets = []
                        self._subifdslevel = -1
                        self._subifds = 0
                        self._ifdindex = 0
                    elif subifds:
                        raise ValueError(
                            'SubIFDs in SubIFDs are not supported'
                        )

                self._datashape = None
                self._colormap = None

            elif compression or packints or tile:
                raise ValueError(
                    'contiguous mode cannot be used with compression or tiles'
                )

            else:
                # consecutive mode
                # write all data, write IFDs/tags later
                self._datashape = (self._datashape[0] + 1,) + datashape
                offset = fh.tell()
                if dataarray is None:
                    fh.write_empty(datasize)
                else:
                    fh.write_array(dataarray, datadtype)
                if returnoffset:
                    return offset, datasize
                return None

        if self._ome is None:
            if description is None:
                self._ome = '.ome.' in fh.extension
            else:
                self._ome = False

        if self._tifffile or self._imagej:
            self._truncate = bool(truncate)
        elif truncate:
            raise ValueError(
                'truncate can only be used with imagej or shaped formats'
            )
        else:
            self._truncate = False

        if self._truncate and (compression or packints or tile):
            raise ValueError(
                'truncate cannot be used with compression, packints, or tiles'
            )

        if datasize == 0:
            # write single placeholder TiffPage for arrays with size=0
            datashape = (0, 0)
            warnings.warn(
                f'{self!r} writing zero-size array to nonconformant TIFF',
                UserWarning,
            )
            # TODO: reconsider this
            # raise ValueError('cannot save zero size array')

        tagnoformat = self.tiff.tagnoformat
        offsetformat = self.tiff.offsetformat
        offsetsize = self.tiff.offsetsize
        tagsize = self.tiff.tagsize

        MINISBLACK = PHOTOMETRIC.MINISBLACK
        MINISWHITE = PHOTOMETRIC.MINISWHITE
        RGB = PHOTOMETRIC.RGB
        YCBCR = PHOTOMETRIC.YCBCR
        PALETTE = PHOTOMETRIC.PALETTE
        CONTIG = PLANARCONFIG.CONTIG
        SEPARATE = PLANARCONFIG.SEPARATE

        # parse input
        if photometric is not None:
            photometric = enumarg(PHOTOMETRIC, photometric)
        if planarconfig:
            planarconfig = enumarg(PLANARCONFIG, planarconfig)
        if extrasamples is not None:
            # TODO: deprecate non-sequence extrasamples
            extrasamples = tuple(
                int(enumarg(EXTRASAMPLE, x)) for x in sequence(extrasamples)
            )

        if compressionargs is None:
            compressionargs = {}

        if compression:
            if isinstance(compression, (tuple, list)):
                # TODO: unreachable
                raise TypeError(
                    "passing multiple values to the 'compression' "
                    "parameter was deprecated in 2022.7.28. "
                    "Use 'compressionargs' to pass extra arguments to the "
                    "compression codec.",
                )
            if isinstance(compression, str):
                compression = compression.upper()
                if compression == 'ZLIB':
                    compression = 8  # ADOBE_DEFLATE
            elif isinstance(compression, bool):
                compression = 8  # ADOBE_DEFLATE
            compressiontag = enumarg(COMPRESSION, compression).value
            compression = True
        else:
            compressiontag = 1
            compression = False

        if compressiontag == 1:
            compressionargs = {}
        elif compressiontag in {33003, 33004, 33005, 34712}:
            # JPEG2000: use J2K instead of JP2
            compressionargs['codecformat'] = 0  # OPJ_CODEC_J2K

        assert compressionargs is not None

        if predictor:
            if not compression:
                raise ValueError('cannot use predictor without compression')
            if compressiontag in TIFF.IMAGE_COMPRESSIONS:
                # don't use predictor with JPEG, JPEG2000, WEBP, PNG, ...
                raise ValueError(
                    'cannot use predictor with '
                    f'{COMPRESSION(compressiontag)!r}'
                )
            if isinstance(predictor, bool):
                if datadtype.kind == 'f':
                    predictortag = 3
                elif datadtype.kind in 'iu' and datadtype.itemsize <= 4:
                    predictortag = 2
                else:
                    raise ValueError(
                        f'cannot use predictor with {datadtype!r}'
                    )
            else:
                predictor = enumarg(PREDICTOR, predictor)
                if (
                    datadtype.kind in 'iu'
                    and predictor.value not in {2, 34892, 34893}
                    and datadtype.itemsize <= 4
                ) or (
                    datadtype.kind == 'f'
                    and predictor.value not in {3, 34894, 34895}
                ):
                    raise ValueError(
                        f'cannot use {predictor!r} with {datadtype!r}'
                    )
                predictortag = predictor.value
        else:
            predictortag = 1

        del predictor
        predictorfunc = TIFF.PREDICTORS[predictortag]

        if self._ome:
            if description is not None:
                warnings.warn(
                    f'{self!r} not writing description to OME-TIFF',
                    UserWarning,
                )
                description = None
            if self._omexml is None:
                if metadata is None:
                    self._omexml = OmeXml()
                else:
                    self._omexml = OmeXml(**metadata)
            if volumetric or (tile and len(tile) > 2):
                raise ValueError('OME-TIFF does not support ImageDepth')
            volumetric = False

        elif self._imagej:
            # if tile is not None or predictor or compression:
            #     warnings.warn(
            #         f'{self!r} the ImageJ format does not support '
            #         'tiles, predictors, compression'
            #     )
            if description is not None:
                warnings.warn(
                    f'{self!r} not writing description to ImageJ file',
                    UserWarning,
                )
                description = None
            if datadtype.char not in 'BHhf':
                raise ValueError(
                    'the ImageJ format does not support data type '
                    f'{datadtype.char!r}'
                )
            if volumetric or (tile and len(tile) > 2):
                raise ValueError(
                    'the ImageJ format does not support ImageDepth'
                )
            volumetric = False
            ijrgb = photometric == RGB if photometric else None
            if datadtype.char != 'B':
                if photometric == RGB:
                    raise ValueError(
                        'the ImageJ format does not support '
                        f'data type {datadtype!r} for RGB'
                    )
                ijrgb = False
            if colormap is not None:
                ijrgb = False
            if metadata is None:
                axes = None
            else:
                axes = metadata.get('axes', None)
            ijshape = imagej_shape(datashape, rgb=ijrgb, axes=axes)
            if planarconfig == SEPARATE:
                raise ValueError(
                    'the ImageJ format does not support planar samples'
                )
            if ijshape[-1] in {3, 4}:
                photometric = RGB
            elif photometric is None:
                if colormap is not None and datadtype.char == 'B':
                    photometric = PALETTE
                else:
                    photometric = MINISBLACK
                planarconfig = None
            planarconfig = CONTIG if ijrgb else None

        # verify colormap and indices
        if colormap is not None:
            colormap = numpy.asarray(colormap, dtype=byteorder + 'H')
            self._colormap = colormap
            if self._imagej:
                if colormap.shape != (3, 256):
                    raise ValueError('invalid colormap shape for ImageJ')
                if datadtype.char == 'B' and photometric in {
                    MINISBLACK,
                    MINISWHITE,
                }:
                    photometric = PALETTE
                elif not (
                    (datadtype.char == 'B' and photometric == PALETTE)
                    or (
                        datadtype.char in 'Hf'
                        and photometric in {MINISBLACK, MINISWHITE}
                    )
                ):
                    warnings.warn(
                        f'{self!r} not writing colormap to ImageJ image with '
                        f'dtype={datadtype} and {photometric=}',
                        UserWarning,
                    )
                    colormap = None
            elif photometric is None and datadtype.char in 'BH':
                photometric = PALETTE
                planarconfig = None
                if colormap.shape != (3, 2 ** (datadtype.itemsize * 8)):
                    raise ValueError('invalid colormap shape')
            elif photometric == PALETTE:
                planarconfig = None
                if datadtype.char not in 'BH':
                    raise ValueError('invalid data dtype for palette-image')
                if colormap.shape != (3, 2 ** (datadtype.itemsize * 8)):
                    raise ValueError('invalid colormap shape')
            else:
                warnings.warn(
                    f'{self!r} not writing colormap with image of '
                    f'dtype={datadtype} and {photometric=}',
                    UserWarning,
                )
                colormap = None

        if tile:
            # verify tile shape

            if (
                not 1 < len(tile) < 4
                or tile[-1] % 16
                or tile[-2] % 16
                or any(i < 1 for i in tile)
            ):
                raise ValueError(f'invalid tile shape {tile}')
            tile = tuple(int(i) for i in tile)
            if volumetric and len(tile) == 2:
                tile = (1,) + tile
            volumetric = len(tile) == 3
        else:
            tile = ()
            volumetric = bool(volumetric)
        assert isinstance(tile, tuple)  # for mypy

        # normalize data shape to 5D or 6D, depending on volume:
        #   (pages, separate_samples, [depth,] length, width, contig_samples)
        shape = reshape_nd(
            datashape,
            TIFF.PHOTOMETRIC_SAMPLES.get(
                photometric, 2  # type: ignore[arg-type]
            ),
        )
        ndim = len(shape)

        if volumetric and ndim < 3:
            volumetric = False

        if photometric is None:
            deprecate = False
            photometric = MINISBLACK
            if bilevel:
                photometric = MINISWHITE
            elif planarconfig == CONTIG:
                if ndim > 2 and shape[-1] in {3, 4}:
                    photometric = RGB
                    deprecate = datadtype.char not in 'BH'
            elif planarconfig == SEPARATE:
                if volumetric and ndim > 3 and shape[-4] in {3, 4}:
                    photometric = RGB
                    deprecate = True
                elif ndim > 2 and shape[-3] in {3, 4}:
                    photometric = RGB
                    deprecate = True
            elif ndim > 2 and shape[-1] in {3, 4}:
                photometric = RGB
                planarconfig = CONTIG
                deprecate = datadtype.char not in 'BH'
            elif self._imagej or self._ome:
                photometric = MINISBLACK
                planarconfig = None
            elif volumetric and ndim > 3 and shape[-4] in {3, 4}:
                photometric = RGB
                planarconfig = SEPARATE
                deprecate = True
            elif ndim > 2 and shape[-3] in {3, 4}:
                photometric = RGB
                planarconfig = SEPARATE
                deprecate = True

            if deprecate:
                if planarconfig == CONTIG:
                    msg = 'contiguous samples', 'parameter is'
                else:
                    msg = (
                        'separate component planes',
                        "and 'planarconfig' parameters are",
                    )
                warnings.warn(
                    f"<tifffile.TiffWriter.write> data with shape {datashape} "
                    f"and dtype '{datadtype}' are stored as RGB with {msg[0]}."
                    " Future versions will store such data as MINISBLACK in "
                    "separate pages by default, unless the 'photometric' "
                    f"{msg[1]} specified.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                del msg
            del deprecate

        del datashape
        assert photometric is not None
        photometricsamples = TIFF.PHOTOMETRIC_SAMPLES[photometric]

        if planarconfig and len(shape) <= (3 if volumetric else 2):
            # TODO: raise error?
            planarconfig = None
            if photometricsamples > 1:
                photometric = MINISBLACK

        if photometricsamples > 1:
            if len(shape) < 3:
                raise ValueError(f'not a {photometric!r} image')
            if len(shape) < 4:
                volumetric = False
            if planarconfig is None:
                if photometric == RGB:
                    samples_set = {photometricsamples, 4}  # allow common alpha
                else:
                    samples_set = {photometricsamples}
                if shape[-1] in samples_set:
                    planarconfig = CONTIG
                elif shape[-4 if volumetric else -3] in samples_set:
                    planarconfig = SEPARATE
                elif shape[-1] > shape[-4 if volumetric else -3]:
                    # TODO: deprecated this?
                    planarconfig = SEPARATE
                else:
                    planarconfig = CONTIG
            if planarconfig == CONTIG:
                storedshape.contig_samples = shape[-1]
                storedshape.width = shape[-2]
                storedshape.length = shape[-3]
                if volumetric:
                    storedshape.depth = shape[-4]
            else:
                storedshape.width = shape[-1]
                storedshape.length = shape[-2]
                if volumetric:
                    storedshape.depth = shape[-3]
                    storedshape.separate_samples = shape[-4]
                else:
                    storedshape.separate_samples = shape[-3]
            if storedshape.samples > photometricsamples:
                storedshape.extrasamples = (
                    storedshape.samples - photometricsamples
                )

        elif photometric == PHOTOMETRIC.CFA:
            if len(shape) != 2:
                raise ValueError('invalid CFA image')
            volumetric = False
            planarconfig = None
            storedshape.width = shape[-1]
            storedshape.length = shape[-2]
            # if all(et[0] != 50706 for et in extratags):
            #     raise ValueError('must specify DNG tags for CFA image')

        elif planarconfig and len(shape) > (3 if volumetric else 2):
            if planarconfig == CONTIG:
                if extrasamples is None or len(extrasamples) > 0:
                    # use extrasamples
                    storedshape.contig_samples = shape[-1]
                    storedshape.width = shape[-2]
                    storedshape.length = shape[-3]
                    if volumetric:
                        storedshape.depth = shape[-4]
                else:
                    planarconfig = None
                    storedshape.contig_samples = 1
                    storedshape.width = shape[-1]
                    storedshape.length = shape[-2]
                    if volumetric:
                        storedshape.depth = shape[-3]
            else:
                storedshape.width = shape[-1]
                storedshape.length = shape[-2]
                if extrasamples is None or len(extrasamples) > 0:
                    # use extrasamples
                    if volumetric:
                        storedshape.depth = shape[-3]
                        storedshape.separate_samples = shape[-4]
                    else:
                        storedshape.separate_samples = shape[-3]
                else:
                    planarconfig = None
                    storedshape.separate_samples = 1
                    if volumetric:
                        storedshape.depth = shape[-3]
            storedshape.extrasamples = storedshape.samples - 1

        else:
            # photometricsamples == 1
            planarconfig = None
            if self._tifffile and (metadata or metadata == {}):
                # remove trailing 1s in shaped series
                while len(shape) > 2 and shape[-1] == 1:
                    shape = shape[:-1]
            elif self._imagej and len(shape) > 2 and shape[-1] == 1:
                # TODO: remove this and sync with ImageJ shape
                shape = shape[:-1]
            if len(shape) < 3:
                volumetric = False
            if not extrasamples:
                storedshape.width = shape[-1]
                storedshape.length = shape[-2]
                if volumetric:
                    storedshape.depth = shape[-3]
            else:
                storedshape.contig_samples = shape[-1]
                storedshape.width = shape[-2]
                storedshape.length = shape[-3]
                if volumetric:
                    storedshape.depth = shape[-4]
                storedshape.extrasamples = storedshape.samples - 1

        if not volumetric and tile and len(tile) == 3 and tile[0] > 1:
            raise ValueError(
                f'<tifffile.TiffWriter.write> cannot write {storedshape!r} '
                f'using volumetric tiles {tile}'
            )

        if subfiletype is not None and subfiletype & 0b100:
            # FILETYPE_MASK
            if not (
                bilevel
                and storedshape.samples == 1
                and photometric in {0, 1, 4}
            ):
                raise ValueError('invalid SubfileType MASK')
            photometric = PHOTOMETRIC.MASK

        packints = False
        if bilevel:
            if bitspersample is not None and bitspersample != 1:
                raise ValueError(f'{bitspersample=} must be 1 for bilevel')
            bitspersample = 1
        elif compressiontag in {6, 7, 34892, 33007}:
            # JPEG
            # TODO: add bitspersample to compressionargs?
            if bitspersample is None:
                if 'bitspersample' in compressionargs:
                    bitspersample = compressionargs['bitspersample']
                else:
                    bitspersample = 12 if datadtype == 'uint16' else 8
            if not 2 <= bitspersample <= 16:
                raise ValueError(
                    f'{bitspersample=} invalid for JPEG compression'
                )
        elif compressiontag in {33003, 33004, 33005, 34712, 50002, 52546}:
            # JPEG2K, JPEGXL
            # TODO: unify with JPEG?
            if bitspersample is None:
                if 'bitspersample' in compressionargs:
                    bitspersample = compressionargs['bitspersample']
                else:
                    bitspersample = datadtype.itemsize * 8
            if not (
                bitspersample > {1: 0, 2: 8, 4: 16}[datadtype.itemsize]
                and bitspersample <= datadtype.itemsize * 8
            ):
                raise ValueError(
                    f'{bitspersample=} out of range of {datadtype=}'
                )
        elif bitspersample is None:
            bitspersample = datadtype.itemsize * 8
        elif (
            datadtype.kind != 'u' or datadtype.itemsize > 4
        ) and bitspersample != datadtype.itemsize * 8:
            raise ValueError(f'{bitspersample=} does not match {datadtype=}')
        elif not (
            bitspersample > {1: 0, 2: 8, 4: 16}[datadtype.itemsize]
            and bitspersample <= datadtype.itemsize * 8
        ):
            raise ValueError(f'{bitspersample=} out of range of {datadtype=}')
        elif compression:
            if bitspersample != datadtype.itemsize * 8:
                raise ValueError(
                    f'{bitspersample=} cannot be used with compression'
                )
        elif bitspersample != datadtype.itemsize * 8:
            packints = True

        if storedshape.frames == -1:
            s0 = storedshape.page_size
            storedshape.frames = 1 if s0 == 0 else product(inputshape) // s0

        if datasize > 0 and not storedshape.is_valid:
            raise RuntimeError(f'invalid {storedshape!r}')

        if photometric == PALETTE:
            if storedshape.samples != 1 or storedshape.extrasamples > 0:
                raise ValueError(f'invalid {storedshape!r} for palette mode')
        elif storedshape.samples < photometricsamples:
            raise ValueError(
                f'not enough samples for {photometric!r}: '
                f'expected {photometricsamples}, got {storedshape.samples}'
            )

        if (
            planarconfig is not None
            and storedshape.planarconfig != planarconfig
        ):
            raise ValueError(
                f'{planarconfig!r} does not match {storedshape!r}'
            )
        del planarconfig

        if dataarray is not None:
            dataarray = dataarray.reshape(storedshape.shape)

        tags = []  # list of (code, ifdentry, ifdvalue, writeonce)

        if tile:
            tagbytecounts = 325  # TileByteCounts
            tagoffsets = 324  # TileOffsets
        else:
            tagbytecounts = 279  # StripByteCounts
            tagoffsets = 273  # StripOffsets
        self._dataoffsetstag = tagoffsets

        pack = self._pack
        addtag = self._addtag

        if extratags is None:
            extratags = ()

        if description is not None:
            # ImageDescription: user provided description
            addtag(tags, 270, 2, 0, description, True)

        # write shape and metadata to ImageDescription
        self._metadata = {} if not metadata else metadata.copy()
        if self._omexml is not None:
            if len(self._omexml.images) == 0:
                # rewritten later at end of file
                description = '\x00\x00\x00\x00'
            else:
                description = None
        elif self._imagej:
            ijmetadata = parse_kwargs(
                self._metadata,
                'Info',
                'Labels',
                'Ranges',
                'LUTs',
                'Plot',
                'ROI',
                'Overlays',
                'Properties',
                'info',
                'labels',
                'ranges',
                'luts',
                'plot',
                'roi',
                'overlays',
                'prop',
            )

            for t in imagej_metadata_tag(ijmetadata, byteorder):
                addtag(tags, *t)
            description = imagej_description(
                inputshape,
                rgb=storedshape.contig_samples in {3, 4},
                colormaped=self._colormap is not None,
                **self._metadata,
            )
            description += '\x00' * 64  # add buffer for in-place update
        elif self._tifffile and (metadata or metadata == {}):
            if self._truncate:
                self._metadata.update(truncated=True)
            description = shaped_description(inputshape, **self._metadata)
            description += '\x00' * 16  # add buffer for in-place update
        # elif metadata is None and self._truncate:
        #     raise ValueError('cannot truncate without writing metadata')
        elif description is not None:
            if not isinstance(description, bytes):
                description = description.encode('ascii')
            self._descriptiontag = TiffTag(
                self, 0, 270, 2, len(description), description, 0
            )
            description = None

        if description is None:
            # disable shaped format if user disabled metadata
            self._tifffile = False
        else:
            description = description.encode('ascii')
            addtag(tags, 270, 2, 0, description, True)
            self._descriptiontag = TiffTag(
                self, 0, 270, 2, len(description), description, 0
            )
        del description

        if software is None:
            software = 'tifffile.py'
        if software:
            addtag(tags, 305, 2, 0, software, True)
        if datetime:
            if isinstance(datetime, str):
                if len(datetime) != 19 or datetime[16] != ':':
                    raise ValueError('invalid datetime string')
            elif isinstance(datetime, DateTime):
                datetime = datetime.strftime('%Y:%m:%d %H:%M:%S')
            else:
                datetime = DateTime.now().strftime('%Y:%m:%d %H:%M:%S')
            addtag(tags, 306, 2, 0, datetime, True)
        addtag(tags, 259, 3, 1, compressiontag)  # Compression
        if compressiontag == 34887:
            # LERC
            if 'compression' not in compressionargs:
                lerc_compression = 0
            elif compressionargs['compression'] is None:
                lerc_compression = 0
            elif compressionargs['compression'] == 'deflate':
                lerc_compression = 1
            elif compressionargs['compression'] == 'zstd':
                lerc_compression = 2
            else:
                raise ValueError(
                    'invalid LERC compression '
                    f'{compressionargs["compression"]!r}'
                )
            addtag(tags, 50674, 4, 2, (4, lerc_compression))
            del lerc_compression
        if predictortag != 1:
            addtag(tags, 317, 3, 1, predictortag)
        addtag(tags, 256, 4, 1, storedshape.width)  # ImageWidth
        addtag(tags, 257, 4, 1, storedshape.length)  # ImageLength
        if tile:
            addtag(tags, 322, 4, 1, tile[-1])  # TileWidth
            addtag(tags, 323, 4, 1, tile[-2])  # TileLength
        if volumetric:
            addtag(tags, 32997, 4, 1, storedshape.depth)  # ImageDepth
            if tile:
                addtag(tags, 32998, 4, 1, tile[0])  # TileDepth
        if subfiletype is not None:
            addtag(tags, 254, 4, 1, subfiletype)  # NewSubfileType
        if (subifds or self._subifds) and self._subifdslevel < 0:
            if self._subifds:
                subifds = self._subifds
            elif hasattr(subifds, '__len__'):
                # allow TiffPage.subifds tuple
                subifds = len(subifds)  # type: ignore[arg-type]
            else:
                subifds = int(subifds)  # type: ignore[arg-type]
            self._subifds = subifds
            addtag(
                tags, 330, 18 if offsetsize > 4 else 13, subifds, [0] * subifds
            )
        if not bilevel and not datadtype.kind == 'u':
            # SampleFormat
            sampleformat = {'u': 1, 'i': 2, 'f': 3, 'c': 6}[datadtype.kind]
            addtag(
                tags,
                339,
                3,
                storedshape.samples,
                (sampleformat,) * storedshape.samples,
            )
        if colormap is not None:
            addtag(tags, 320, 3, colormap.size, colormap)
        if iccprofile is not None:
            addtag(tags, 34675, 7, len(iccprofile), iccprofile)
        addtag(tags, 277, 3, 1, storedshape.samples)
        if bilevel:
            # PlanarConfiguration
            if storedshape.samples > 1:
                addtag(tags, 284, 3, 1, storedshape.planarconfig)
        elif storedshape.samples > 1:
            # PlanarConfiguration
            addtag(tags, 284, 3, 1, storedshape.planarconfig)
            # BitsPerSample
            addtag(
                tags,
                258,
                3,
                storedshape.samples,
                (bitspersample,) * storedshape.samples,
            )
        else:
            addtag(tags, 258, 3, 1, bitspersample)
        if storedshape.extrasamples > 0:
            if extrasamples is not None:
                if storedshape.extrasamples != len(extrasamples):
                    raise ValueError(
                        'wrong number of extrasamples '
                        f'{storedshape.extrasamples} != {len(extrasamples)}'
                    )
                addtag(tags, 338, 3, len(extrasamples), extrasamples)
            elif photometric == RGB and storedshape.extrasamples == 1:
                # Unassociated alpha channel
                addtag(tags, 338, 3, 1, 2)
            else:
                # Unspecified alpha channel
                addtag(
                    tags,
                    338,
                    3,
                    storedshape.extrasamples,
                    (0,) * storedshape.extrasamples,
                )

        if jpegtables is not None:
            addtag(tags, 347, 7, len(jpegtables), jpegtables)

        if (
            compressiontag == 7
            and storedshape.planarconfig == 1
            and photometric in {RGB, YCBCR}
        ):
            # JPEG compression with subsampling
            # TODO: use JPEGTables for multiple tiles or strips
            if subsampling is None:
                subsampling = (2, 2)
            elif subsampling not in {(1, 1), (2, 1), (2, 2), (4, 1)}:
                raise ValueError(
                    f'invalid subsampling factors {subsampling!r}'
                )
            maxsampling = max(subsampling) * 8
            if tile and (tile[-1] % maxsampling or tile[-2] % maxsampling):
                raise ValueError(f'tile shape not a multiple of {maxsampling}')
            if storedshape.extrasamples > 1:
                raise ValueError('JPEG subsampling requires RGB(A) images')
            addtag(tags, 530, 3, 2, subsampling)  # YCbCrSubSampling
            # use PhotometricInterpretation YCBCR by default
            outcolorspace = enumarg(
                PHOTOMETRIC, compressionargs.get('outcolorspace', 6)
            )
            compressionargs['subsampling'] = subsampling
            compressionargs['colorspace'] = photometric.name
            compressionargs['outcolorspace'] = outcolorspace.name
            addtag(tags, 262, 3, 1, outcolorspace)
            if outcolorspace == YCBCR:
                # ReferenceBlackWhite is required for YCBCR
                if all(et[0] != 532 for et in extratags):
                    addtag(
                        tags,
                        532,
                        5,
                        6,
                        (0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1),
                    )
        else:
            if subsampling not in {None, (1, 1)}:
                logger().warning(
                    f'{self!r} cannot apply subsampling {subsampling!r}'
                )
            subsampling = None
            maxsampling = 1
            addtag(
                tags, 262, 3, 1, photometric.value
            )  # PhotometricInterpretation
            if photometric == YCBCR:
                # YCbCrSubSampling and ReferenceBlackWhite
                addtag(tags, 530, 3, 2, (1, 1))
                if all(et[0] != 532 for et in extratags):
                    addtag(
                        tags,
                        532,
                        5,
                        6,
                        (0, 1, 255, 1, 128, 1, 255, 1, 128, 1, 255, 1),
                    )

        if resolutionunit is not None:
            resolutionunit = enumarg(RESUNIT, resolutionunit)
        elif self._imagej or resolution is None:
            resolutionunit = RESUNIT.NONE
        else:
            resolutionunit = RESUNIT.INCH

        if resolution is not None:
            addtag(tags, 282, 5, 1, rational(resolution[0]))  # XResolution
            addtag(tags, 283, 5, 1, rational(resolution[1]))  # YResolution
            if len(resolution) > 2:
                # TODO: unreachable
                raise ValueError(
                    "passing a unit along with the 'resolution' parameter "
                    "was deprecated in 2022.7.28. "
                    "Use the 'resolutionunit' parameter.",
                )
            addtag(tags, 296, 3, 1, resolutionunit)  # ResolutionUnit
        else:
            addtag(tags, 282, 5, 1, (1, 1))  # XResolution
            addtag(tags, 283, 5, 1, (1, 1))  # YResolution
            addtag(tags, 296, 3, 1, resolutionunit)  # ResolutionUnit

        # can save data array contiguous
        contiguous = not (compression or packints or bilevel)
        if tile:
            # one chunk per tile per plane
            if len(tile) == 2:
                tiles = (
                    (storedshape.length + tile[0] - 1) // tile[0],
                    (storedshape.width + tile[1] - 1) // tile[1],
                )
                contiguous = (
                    contiguous
                    and storedshape.length == tile[0]
                    and storedshape.width == tile[1]
                )
            else:
                tiles = (
                    (storedshape.depth + tile[0] - 1) // tile[0],
                    (storedshape.length + tile[1] - 1) // tile[1],
                    (storedshape.width + tile[2] - 1) // tile[2],
                )
                contiguous = (
                    contiguous
                    and storedshape.depth == tile[0]
                    and storedshape.length == tile[1]
                    and storedshape.width == tile[2]
                )
            numtiles = product(tiles) * storedshape.separate_samples
            databytecounts = [
                product(tile) * storedshape.contig_samples * datadtype.itemsize
            ] * numtiles
            bytecountformat = self._bytecount_format(
                databytecounts, compressiontag
            )
            addtag(
                tags, tagbytecounts, bytecountformat, numtiles, databytecounts
            )
            addtag(tags, tagoffsets, offsetformat, numtiles, [0] * numtiles)
            bytecountformat = f'{numtiles}{bytecountformat}'
            if not contiguous:
                if dataarray is not None:
                    dataiter = iter_tiles(dataarray, tile, tiles)
                elif dataiter is None and not (
                    compression or packints or bilevel
                ):

                    def dataiter_(
                        numtiles: int = numtiles * storedshape.frames,
                        bytecount: int = databytecounts[0],
                    ) -> Iterator[bytes]:
                        # yield empty tiles
                        chunk = bytes(bytecount)
                        for _ in range(numtiles):
                            yield chunk

                    dataiter = dataiter_()

            rowsperstrip = 0

        elif contiguous and (
            rowsperstrip is None or rowsperstrip >= storedshape.length
        ):
            count = storedshape.separate_samples * storedshape.depth
            databytecounts = [
                storedshape.length
                * storedshape.width
                * storedshape.contig_samples
                * datadtype.itemsize
            ] * count
            bytecountformat = self._bytecount_format(
                databytecounts, compressiontag
            )
            addtag(tags, tagbytecounts, bytecountformat, count, databytecounts)
            addtag(tags, tagoffsets, offsetformat, count, [0] * count)
            addtag(tags, 278, 4, 1, storedshape.length)  # RowsPerStrip
            bytecountformat = f'{count}{bytecountformat}'
            rowsperstrip = storedshape.length
            numstrips = count

        else:
            # use rowsperstrip
            rowsize = (
                storedshape.width
                * storedshape.contig_samples
                * datadtype.itemsize
            )
            if compressiontag == 48124:
                # Jetraw works on whole camera frame
                rowsperstrip = storedshape.length
            if rowsperstrip is None:
                # compress ~256 KB chunks by default
                # TIFF-EP requires <= 64 KB
                if compression:
                    rowsperstrip = 262144 // rowsize
                else:
                    rowsperstrip = storedshape.length
            if rowsperstrip < 1:
                rowsperstrip = maxsampling
            elif rowsperstrip > storedshape.length:
                rowsperstrip = storedshape.length
            elif subsampling and rowsperstrip % maxsampling:
                rowsperstrip = (
                    math.ceil(rowsperstrip / maxsampling) * maxsampling
                )
            assert rowsperstrip is not None
            addtag(tags, 278, 4, 1, rowsperstrip)  # RowsPerStrip

            numstrips1 = (
                storedshape.length + rowsperstrip - 1
            ) // rowsperstrip
            numstrips = (
                numstrips1 * storedshape.separate_samples * storedshape.depth
            )
            # TODO: save bilevel data with rowsperstrip
            stripsize = rowsperstrip * rowsize
            databytecounts = [stripsize] * numstrips
            laststripsize = stripsize - rowsize * (
                numstrips1 * rowsperstrip - storedshape.length
            )
            for i in range(numstrips1 - 1, numstrips, numstrips1):
                databytecounts[i] = laststripsize
            bytecountformat = self._bytecount_format(
                databytecounts, compressiontag
            )
            addtag(
                tags, tagbytecounts, bytecountformat, numstrips, databytecounts
            )
            addtag(tags, tagoffsets, offsetformat, numstrips, [0] * numstrips)
            bytecountformat = bytecountformat * numstrips

            if dataarray is not None and not contiguous:
                dataiter = iter_images(dataarray)

        if dataiter is None and not contiguous:
            raise ValueError('cannot write non-contiguous empty file')

        # add extra tags from user; filter duplicate and select tags
        extratag: TagTuple
        tagset = {t[0] for t in tags}
        tagset.update(TIFF.TAG_FILTERED)
        for extratag in extratags:
            if extratag[0] in tagset:
                logger().warning(
                    f'{self!r} not writing extratag {extratag[0]}'
                )
            else:
                addtag(tags, *extratag)
        del tagset
        del extratags

        # TODO: check TIFFReadDirectoryCheckOrder warning in files containing
        #   multiple tags of same code
        # the entries in an IFD must be sorted in ascending order by tag code
        tags = sorted(tags, key=lambda x: x[0])

        # define compress function
        compressionaxis: int = -2
        bytesiter: bool = False

        iteritem: NDArray[Any] | bytes | None
        if dataiter is not None:
            iteritem, dataiter = peek_iterator(dataiter)
            bytesiter = isinstance(iteritem, bytes)
            if not bytesiter:
                iteritem = numpy.asarray(iteritem)
                if (
                    tile
                    and storedshape.contig_samples == 1
                    and iteritem.shape[-1] != 1
                ):
                    # issue 185
                    compressionaxis = -1
                if iteritem.dtype.char != datadtype.char:
                    raise ValueError(
                        f'dtype of iterator {iteritem.dtype!r} '
                        f'does not match dtype {datadtype!r}'
                    )
        else:
            iteritem = None

        if bilevel:
            if compressiontag == 1:

                def compressionfunc1(
                    data: Any, axis: int = compressionaxis
                ) -> bytes:
                    return numpy.packbits(data, axis=axis).tobytes()

                compressionfunc = compressionfunc1

            elif compressiontag in {5, 32773, 8, 32946, 50013, 34925, 50000}:
                # LZW, PackBits, deflate, LZMA, ZSTD
                def compressionfunc2(
                    data: Any,
                    compressor: Any = TIFF.COMPRESSORS[compressiontag],
                    axis: int = compressionaxis,
                    kwargs: Any = compressionargs,
                ) -> bytes:
                    data = numpy.packbits(data, axis=axis).tobytes()
                    return compressor(data, **kwargs)

                compressionfunc = compressionfunc2

            else:
                raise NotImplementedError('cannot compress bilevel image')

        elif compression:
            compressor = TIFF.COMPRESSORS[compressiontag]

            if compressiontag == 32773:
                # PackBits
                compressionargs['axis'] = compressionaxis

            # elif compressiontag == 48124:
            #     # Jetraw
            #     imagecodecs.jetraw_init(
            #         parameters=compressionargs.pop('parameters', None),
            #         verbose=compressionargs.pop('verbose', None),
            #     )
            #     if not 'identifier' in compressionargs:
            #         raise ValueError(
            #             "jetraw_encode() missing argument: 'identifier'"
            #         )

            if subsampling:
                # JPEG with subsampling
                def compressionfunc(
                    data: Any,
                    compressor: Any = compressor,
                    kwargs: Any = compressionargs,
                ) -> bytes:
                    return compressor(data, **kwargs)

            elif predictorfunc is not None:

                def compressionfunc(
                    data: Any,
                    predictorfunc: Any = predictorfunc,
                    compressor: Any = compressor,
                    axis: int = compressionaxis,
                    kwargs: Any = compressionargs,
                ) -> bytes:
                    data = predictorfunc(data, axis=axis)
                    return compressor(data, **kwargs)

            elif compressionargs:

                def compressionfunc(
                    data: Any,
                    compressor: Any = compressor,
                    kwargs: Any = compressionargs,
                ) -> bytes:
                    return compressor(data, **kwargs)

            elif compressiontag > 1:
                compressionfunc = compressor

            else:
                compressionfunc = None

        elif packints:

            def compressionfunc(
                data: Any,
                bps: Any = bitspersample,
                axis: int = compressionaxis,
            ) -> bytes:
                return imagecodecs.packints_encode(data, bps, axis=axis)

        else:
            compressionfunc = None

        del compression
        if not contiguous and not bytesiter and compressionfunc is not None:
            # create iterator of encoded tiles or strips
            bytesiter = True
            if tile:
                # dataiter yields tiles
                tileshape = tile + (storedshape.contig_samples,)
                tilesize = product(tileshape) * datadtype.itemsize
                maxworkers = TiffWriter._maxworkers(
                    maxworkers,
                    numtiles * storedshape.frames,
                    tilesize,
                    compressiontag,
                )
                # yield encoded tiles
                dataiter = encode_chunks(
                    numtiles * storedshape.frames,
                    dataiter,  # type: ignore[arg-type]
                    compressionfunc,
                    tileshape,
                    datadtype,
                    maxworkers,
                    buffersize,
                    True,
                )
            else:
                # dataiter yields frames
                maxworkers = TiffWriter._maxworkers(
                    maxworkers,
                    numstrips * storedshape.frames,
                    stripsize,
                    compressiontag,
                )
                # yield strips
                dataiter = iter_strips(
                    dataiter,  # type: ignore[arg-type]
                    storedshape.page_shape,
                    datadtype,
                    rowsperstrip,
                )
                # yield encoded strips
                dataiter = encode_chunks(
                    numstrips * storedshape.frames,
                    dataiter,
                    compressionfunc,
                    (
                        rowsperstrip,
                        storedshape.width,
                        storedshape.contig_samples,
                    ),
                    datadtype,
                    maxworkers,
                    buffersize,
                    False,
                )

        fhpos = fh.tell()
        # commented out to allow image data beyond 4GB in classic TIFF
        # if (
        #     not (
        #         offsetsize > 4
        #         or self._imagej or compressionfunc is not None
        #     )
        #     and fhpos + datasize > 2**32 - 1
        # ):
        #     raise ValueError('data too large for classic TIFF format')

        dataoffset: int = 0

        # if not compressed or multi-tiled, write the first IFD and then
        # all data contiguously; else, write all IFDs and data interleaved
        for pageindex in range(1 if contiguous else storedshape.frames):
            ifdpos = fhpos
            if ifdpos % 2:
                # position of IFD must begin on a word boundary
                fh.write(b'\x00')
                ifdpos += 1

            if self._subifdslevel < 0:
                # update pointer at ifdoffset
                fh.seek(self._ifdoffset)
                fh.write(pack(offsetformat, ifdpos))

            fh.seek(ifdpos)

            # create IFD in memory
            if pageindex < 2:
                subifdsoffsets = None
                ifd = io.BytesIO()
                ifd.write(pack(tagnoformat, len(tags)))
                tagoffset = ifd.tell()
                ifd.write(b''.join(t[1] for t in tags))
                ifdoffset = ifd.tell()
                ifd.write(pack(offsetformat, 0))  # offset to next IFD
                # write tag values and patch offsets in ifdentries
                for tagindex, tag in enumerate(tags):
                    offset = tagoffset + tagindex * tagsize + 4 + offsetsize
                    code = tag[0]
                    value = tag[2]
                    if value:
                        pos = ifd.tell()
                        if pos % 2:
                            # tag value is expected to begin on word boundary
                            ifd.write(b'\x00')
                            pos += 1
                        ifd.seek(offset)
                        ifd.write(pack(offsetformat, ifdpos + pos))
                        ifd.seek(pos)
                        ifd.write(value)
                        if code == tagoffsets:
                            dataoffsetsoffset = offset, pos
                        elif code == tagbytecounts:
                            databytecountsoffset = offset, pos
                        elif code == 270:
                            if (
                                self._descriptiontag is not None
                                and self._descriptiontag.offset == 0
                                and value.startswith(
                                    self._descriptiontag.value
                                )
                            ):
                                self._descriptiontag.offset = (
                                    ifdpos + tagoffset + tagindex * tagsize
                                )
                                self._descriptiontag.valueoffset = ifdpos + pos
                        elif code == 330:
                            subifdsoffsets = offset, pos
                    elif code == tagoffsets:
                        dataoffsetsoffset = offset, None
                    elif code == tagbytecounts:
                        databytecountsoffset = offset, None
                    elif code == 270:
                        if (
                            self._descriptiontag is not None
                            and self._descriptiontag.offset == 0
                            and self._descriptiontag.value in tag[1][-4:]
                        ):
                            self._descriptiontag.offset = (
                                ifdpos + tagoffset + tagindex * tagsize
                            )
                            self._descriptiontag.valueoffset = (
                                self._descriptiontag.offset + offsetsize + 4
                            )
                    elif code == 330:
                        subifdsoffsets = offset, None
                ifdsize = ifd.tell()
                if ifdsize % 2:
                    ifd.write(b'\x00')
                    ifdsize += 1

            # write IFD later when strip/tile bytecounts and offsets are known
            fh.seek(ifdsize, os.SEEK_CUR)

            # write image data
            dataoffset = fh.tell()
            if align is None:
                align = 16
            skip = (align - (dataoffset % align)) % align
            fh.seek(skip, os.SEEK_CUR)
            dataoffset += skip

            if contiguous:
                # write all image data contiguously
                if dataiter is not None:
                    byteswritten = 0
                    if bytesiter:
                        for iteritem in dataiter:
                            # assert isinstance(iteritem, bytes)
                            byteswritten += fh.write(
                                iteritem  # type: ignore[arg-type]
                            )
                            del iteritem
                    else:
                        pagesize = storedshape.page_size * datadtype.itemsize
                        for iteritem in dataiter:
                            if iteritem is None:
                                byteswritten += fh.write_empty(pagesize)
                            else:
                                # assert isinstance(iteritem, numpy.ndarray)
                                byteswritten += fh.write_array(
                                    iteritem,  # type: ignore[arg-type]
                                    datadtype,
                                )
                            del iteritem
                    if byteswritten != datasize:
                        raise ValueError(
                            'iterator contains wrong number of bytes '
                            f'{byteswritten} != {datasize}'
                        )
                elif dataarray is None:
                    fh.write_empty(datasize)
                else:
                    fh.write_array(dataarray, datadtype)

            elif bytesiter:
                # write tiles or strips
                assert dataiter is not None
                for chunkindex in range(numtiles if tile else numstrips):
                    iteritem = cast(bytes, next(dataiter))
                    # assert isinstance(iteritem, bytes)
                    databytecounts[chunkindex] = len(iteritem)
                    fh.write(iteritem)
                    del iteritem

            elif tile:
                # write uncompressed tiles
                assert dataiter is not None
                tileshape = tile + (storedshape.contig_samples,)
                tilesize = product(tileshape) * datadtype.itemsize
                for tileindex in range(numtiles):
                    iteritem = next(dataiter)
                    if iteritem is None:
                        databytecounts[tileindex] = 0
                        # fh.write_empty(tilesize)
                        continue
                    # assert not isinstance(iteritem, bytes)
                    iteritem = numpy.ascontiguousarray(iteritem, datadtype)
                    if iteritem.nbytes != tilesize:
                        # if iteritem.dtype != datadtype:
                        #     raise ValueError(
                        #         'dtype of tile does not match data'
                        #     )
                        if iteritem.nbytes > tilesize:
                            raise ValueError('tile is too large')
                        pad = tuple(
                            (0, i - j)
                            for i, j in zip(tileshape, iteritem.shape)
                        )
                        iteritem = numpy.pad(iteritem, pad)
                    fh.write_array(iteritem)
                    del iteritem

            else:
                raise RuntimeError('unreachable code')

            # update strip/tile offsets
            assert dataoffsetsoffset is not None
            offset, pos = dataoffsetsoffset
            ifd.seek(offset)
            if pos is not None:
                ifd.write(pack(offsetformat, ifdpos + pos))
                ifd.seek(pos)
                offset = dataoffset
                for size in databytecounts:
                    ifd.write(pack(offsetformat, offset if size > 0 else 0))
                    offset += size
            else:
                ifd.write(pack(offsetformat, dataoffset))

            if compressionfunc is not None or (tile and dataarray is None):
                # update strip/tile bytecounts
                assert databytecountsoffset is not None
                offset, pos = databytecountsoffset
                ifd.seek(offset)
                if pos is not None:
                    ifd.write(pack(offsetformat, ifdpos + pos))
                    ifd.seek(pos)
                ifd.write(pack(bytecountformat, *databytecounts))

            if subifdsoffsets is not None:
                # update and save pointer to SubIFDs tag values if necessary
                offset, pos = subifdsoffsets
                if pos is not None:
                    ifd.seek(offset)
                    ifd.write(pack(offsetformat, ifdpos + pos))
                    self._subifdsoffsets.append(ifdpos + pos)
                else:
                    self._subifdsoffsets.append(ifdpos + offset)

            fhpos = fh.tell()
            fh.seek(ifdpos)
            fh.write(ifd.getbuffer())
            fh.flush()

            if self._subifdslevel < 0:
                self._ifdoffset = ifdpos + ifdoffset
            else:
                # update SubIFDs tag values
                fh.seek(
                    self._subifdsoffsets[self._ifdindex]
                    + self._subifdslevel * offsetsize
                )
                fh.write(pack(offsetformat, ifdpos))

                # update SubIFD chain offsets
                if self._subifdslevel == 0:
                    self._nextifdoffsets.append(ifdpos + ifdoffset)
                else:
                    fh.seek(self._nextifdoffsets[self._ifdindex])
                    fh.write(pack(offsetformat, ifdpos))
                    self._nextifdoffsets[self._ifdindex] = ifdpos + ifdoffset
                self._ifdindex += 1
                self._ifdindex %= len(self._subifdsoffsets)

            fh.seek(fhpos)

            # remove tags that should be written only once
            if pageindex == 0:
                tags = [tag for tag in tags if not tag[-1]]

        assert dataoffset > 0

        self._datashape = (1,) + inputshape
        self._datadtype = datadtype
        self._dataoffset = dataoffset
        self._databytecounts = databytecounts
        self._storedshape = storedshape

        if contiguous:
            # write remaining IFDs/tags later
            self._tags = tags
            # return offset and size of image data
            if returnoffset:
                return dataoffset, sum(databytecounts)
        return None

    def overwrite_description(self, description: str, /) -> None:
        """Overwrite value of last ImageDescription tag.

        Can be used to write OME-XML after writing images.
        Ends a contiguous series.

        """
        if self._descriptiontag is None:
            raise ValueError('no ImageDescription tag found')
        self._write_remaining_pages()
        self._descriptiontag.overwrite(description, erase=False)
        self._descriptiontag = None

    def close(self) -> None:
        """Write remaining pages and close file handle."""
        try:
            if not self._truncate:
                self._write_remaining_pages()
            self._write_image_description()
        finally:
            try:
                self._fh.close()
            except Exception:
                pass

    @property
    def filehandle(self) -> FileHandle:
        """File handle to write file."""
        return self._fh

    def _write_remaining_pages(self) -> None:
        """Write outstanding IFDs and tags to file."""
        if not self._tags or self._truncate or self._datashape is None:
            return

        assert self._storedshape is not None
        assert self._databytecounts is not None
        assert self._dataoffset is not None

        pageno: int = self._storedshape.frames * self._datashape[0] - 1
        if pageno < 1:
            self._tags = None
            self._dataoffset = None
            self._databytecounts = None
            return

        fh = self._fh
        fhpos: int = fh.tell()
        if fhpos % 2:
            fh.write(b'\x00')
            fhpos += 1

        pack = struct.pack
        offsetformat: str = self.tiff.offsetformat
        offsetsize: int = self.tiff.offsetsize
        tagnoformat: str = self.tiff.tagnoformat
        tagsize: int = self.tiff.tagsize
        dataoffset: int = self._dataoffset
        pagedatasize: int = sum(self._databytecounts)
        subifdsoffsets: tuple[int, int | None] | None = None
        dataoffsetsoffset: tuple[int, int | None]
        pos: int | None
        offset: int

        # construct template IFD in memory
        # must patch offsets to next IFD and data before writing to file
        ifd = io.BytesIO()
        ifd.write(pack(tagnoformat, len(self._tags)))
        tagoffset = ifd.tell()
        ifd.write(b''.join(t[1] for t in self._tags))
        ifdoffset = ifd.tell()
        ifd.write(pack(offsetformat, 0))  # offset to next IFD
        # tag values
        for tagindex, tag in enumerate(self._tags):
            offset = tagoffset + tagindex * tagsize + offsetsize + 4
            code = tag[0]
            value = tag[2]
            if value:
                pos = ifd.tell()
                if pos % 2:
                    # tag value is expected to begin on word boundary
                    ifd.write(b'\x00')
                    pos += 1
                ifd.seek(offset)
                try:
                    ifd.write(pack(offsetformat, fhpos + pos))
                except Exception as exc:  # struct.error
                    if self._imagej:
                        warnings.warn(
                            f'{self!r} truncating ImageJ file', UserWarning
                        )
                        self._truncate = True
                        return
                    raise ValueError(
                        'data too large for non-BigTIFF file'
                    ) from exc
                ifd.seek(pos)
                ifd.write(value)
                if code == self._dataoffsetstag:
                    # save strip/tile offsets for later updates
                    dataoffsetsoffset = offset, pos
                elif code == 330:
                    # save subifds offsets for later updates
                    subifdsoffsets = offset, pos
            elif code == self._dataoffsetstag:
                dataoffsetsoffset = offset, None
            elif code == 330:
                subifdsoffsets = offset, None

        ifdsize = ifd.tell()
        if ifdsize % 2:
            ifd.write(b'\x00')
            ifdsize += 1

        # check if all IFDs fit in file
        if offsetsize < 8 and fhpos + ifdsize * pageno > 2**32 - 32:
            if self._imagej:
                warnings.warn(f'{self!r} truncating ImageJ file', UserWarning)
                self._truncate = True
                return
            raise ValueError('data too large for non-BigTIFF file')

        # assemble IFD chain in memory from IFD template
        ifds = io.BytesIO(bytes(ifdsize * pageno))
        ifdpos = fhpos
        for _ in range(pageno):
            # update strip/tile offsets in IFD
            dataoffset += pagedatasize  # offset to image data
            offset, pos = dataoffsetsoffset
            ifd.seek(offset)
            if pos is not None:
                ifd.write(pack(offsetformat, ifdpos + pos))
                ifd.seek(pos)
                offset = dataoffset
                for size in self._databytecounts:
                    ifd.write(pack(offsetformat, offset))
                    offset += size
            else:
                ifd.write(pack(offsetformat, dataoffset))

            if subifdsoffsets is not None:
                offset, pos = subifdsoffsets
                self._subifdsoffsets.append(
                    ifdpos + (pos if pos is not None else offset)
                )

            if self._subifdslevel < 0:
                if subifdsoffsets is not None:
                    # update pointer to SubIFDs tag values if necessary
                    offset, pos = subifdsoffsets
                    if pos is not None:
                        ifd.seek(offset)
                        ifd.write(pack(offsetformat, ifdpos + pos))

                # update pointer at ifdoffset to point to next IFD in file
                ifdpos += ifdsize
                ifd.seek(ifdoffset)
                ifd.write(pack(offsetformat, ifdpos))

            else:
                # update SubIFDs tag values in file
                fh.seek(
                    self._subifdsoffsets[self._ifdindex]
                    + self._subifdslevel * offsetsize
                )
                fh.write(pack(offsetformat, ifdpos))

                # update SubIFD chain
                if self._subifdslevel == 0:
                    self._nextifdoffsets.append(ifdpos + ifdoffset)
                else:
                    fh.seek(self._nextifdoffsets[self._ifdindex])
                    fh.write(pack(offsetformat, ifdpos))
                    self._nextifdoffsets[self._ifdindex] = ifdpos + ifdoffset
                self._ifdindex += 1
                self._ifdindex %= len(self._subifdsoffsets)
                ifdpos += ifdsize

            # write IFD entry
            ifds.write(ifd.getbuffer())

        # terminate IFD chain
        ifdoffset += ifdsize * (pageno - 1)
        ifds.seek(ifdoffset)
        ifds.write(pack(offsetformat, 0))
        # write IFD chain to file
        fh.seek(fhpos)
        fh.write(ifds.getbuffer())

        if self._subifdslevel < 0:
            # update file to point to new IFD chain
            pos = fh.tell()
            fh.seek(self._ifdoffset)
            fh.write(pack(offsetformat, fhpos))
            fh.flush()
            fh.seek(pos)
            self._ifdoffset = fhpos + ifdoffset

        self._tags = None
        self._dataoffset = None
        self._databytecounts = None
        # do not reset _storedshape, _datashape, _datadtype

    def _write_image_description(self) -> None:
        """Write metadata to ImageDescription tag."""
        if self._datashape is None or self._descriptiontag is None:
            self._descriptiontag = None
            return

        assert self._storedshape is not None
        assert self._datadtype is not None

        if self._omexml is not None:
            if self._subifdslevel < 0:
                assert self._metadata is not None
                self._omexml.addimage(
                    dtype=self._datadtype,
                    shape=self._datashape[
                        0 if self._datashape[0] != 1 else 1 :
                    ],
                    storedshape=self._storedshape.shape,
                    **self._metadata,
                )
            description = self._omexml.tostring(declaration=True)
        elif self._datashape[0] == 1:
            # description already up-to-date
            self._descriptiontag = None
            return
        # elif self._subifdslevel >= 0:
        #     # don't write metadata to SubIFDs
        #     return
        elif self._imagej:
            assert self._metadata is not None
            colormapped = self._colormap is not None
            isrgb = self._storedshape.samples in {3, 4}
            description = imagej_description(
                self._datashape,
                rgb=isrgb,
                colormaped=colormapped,
                **self._metadata,
            )
        elif not self._tifffile:
            self._descriptiontag = None
            return
        else:
            assert self._metadata is not None
            description = shaped_description(self._datashape, **self._metadata)

        self._descriptiontag.overwrite(description.encode(), erase=False)
        self._descriptiontag = None

    def _addtag(
        self,
        tags: list[tuple[int, bytes, bytes | None, bool]],
        code: int | str,
        dtype: int | str,
        count: int | None,
        value: Any,
        writeonce: bool = False,
        /,
    ) -> None:
        """Append (code, ifdentry, ifdvalue, writeonce) to tags list.

        Compute ifdentry and ifdvalue bytes from code, dtype, count, value.

        """
        pack = self._pack

        if not isinstance(code, int):
            code = TIFF.TAGS[code]
        try:
            datatype = cast(int, dtype)
            dataformat = TIFF.DATA_FORMATS[datatype][-1]
        except KeyError as exc:
            try:
                dataformat = cast(str, dtype)
                if dataformat[0] in '<>':
                    dataformat = dataformat[1:]
                datatype = TIFF.DATA_DTYPES[dataformat]
            except (KeyError, TypeError):
                raise ValueError(f'unknown dtype {dtype}') from exc
        del dtype

        rawcount = count
        if datatype == 2:
            # string
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
            if code == 270:
                rawcount = int(value.find(b'\x00\x00'))
                if rawcount < 0:
                    rawcount = count
                else:
                    # length of string without buffer
                    rawcount = max(self.tiff.offsetsize + 1, rawcount + 1)
                    rawcount = min(count, rawcount)
            else:
                rawcount = count
            value = (value,)

        elif isinstance(value, bytes):
            # packed binary data
            itemsize = struct.calcsize(dataformat)
            if len(value) % itemsize:
                raise ValueError('invalid packed binary data')
            count = len(value) // itemsize
            rawcount = count

        elif count is None:
            raise ValueError('invalid count')
        else:
            count = int(count)

        if datatype in {5, 10}:  # rational
            count *= 2
            dataformat = dataformat[-1]

        ifdentry = [
            pack('HH', code, datatype),
            pack(self.tiff.offsetformat, rawcount),
        ]

        ifdvalue = None
        if struct.calcsize(dataformat) * count <= self.tiff.offsetsize:
            # value(s) can be written directly
            valueformat = f'{self.tiff.offsetsize}s'
            if isinstance(value, bytes):
                ifdentry.append(pack(valueformat, value))
            elif count == 1:
                if isinstance(value, (tuple, list, numpy.ndarray)):
                    value = value[0]
                ifdentry.append(pack(valueformat, pack(dataformat, value)))
            else:
                ifdentry.append(
                    pack(valueformat, pack(f'{count}{dataformat}', *value))
                )
        else:
            # use offset to value(s)
            ifdentry.append(pack(self.tiff.offsetformat, 0))
            if isinstance(value, bytes):
                ifdvalue = value
            elif isinstance(value, numpy.ndarray):
                if value.size != count:
                    raise RuntimeError('value.size != count')
                if value.dtype.char != dataformat:
                    raise RuntimeError('value.dtype.char != dtype')
                ifdvalue = value.tobytes()
            elif isinstance(value, (tuple, list)):
                ifdvalue = pack(f'{count}{dataformat}', *value)
            else:
                ifdvalue = pack(dataformat, value)
        tags.append((code, b''.join(ifdentry), ifdvalue, writeonce))

    def _pack(self, fmt: str, *val: Any) -> bytes:
        """Return values packed to bytes according to format."""
        if fmt[0] not in '<>':
            fmt = self.tiff.byteorder + fmt
        return struct.pack(fmt, *val)

    def _bytecount_format(
        self, bytecounts: Sequence[int], compression: int, /
    ) -> str:
        """Return small bytecount format."""
        if len(bytecounts) == 1:
            return self.tiff.offsetformat[1]
        bytecount = bytecounts[0]
        if compression > 1:
            bytecount = bytecount * 10
        if bytecount < 2**16:
            return 'H'
        if bytecount < 2**32:
            return 'I'
        return self.tiff.offsetformat[1]

    @staticmethod
    def _maxworkers(
        maxworkers: int | None,
        numchunks: int,
        chunksize: int,
        compression: int,
    ) -> int:
        """Return number of threads to encode segments."""
        if maxworkers is not None:
            return maxworkers
        if (
            # imagecodecs is None or
            compression <= 1
            or numchunks < 2
            or chunksize < 1024
            or compression == 48124  # Jetraw is not thread-safe?
        ):
            return 1
        # the following is based on benchmarking RGB tile sizes vs maxworkers
        # using a (8228, 11500, 3) uint8 WSI slide:
        if chunksize < 131072 and compression in {
            7,  # JPEG
            33007,  # ALT_JPG
            34892,  # JPEG_LOSSY
            32773,  # PackBits
            34887,  # LERC
        }:
            return 1
        if chunksize < 32768 and compression in {
            5,  # LZW
            8,  # zlib
            32946,  # zlib
            50000,  # zstd
            50013,  # zlib/pixtiff
        }:
            # zlib,
            return 1
        if chunksize < 8192 and compression in {
            34934,  # JPEG XR
            22610,  # JPEG XR
            34933,  # PNG
        }:
            return 1
        if chunksize < 2048 and compression in {
            33003,  # JPEG2000
            33004,  # JPEG2000
            33005,  # JPEG2000
            34712,  # JPEG2000
            50002,  # JPEG XL
            52546,  # JPEG XL DNG
        }:
            return 1
        if chunksize < 1024 and compression in {
            34925,  # LZMA
            50001,  # WebP
        }:
            return 1
        if compression == 34887:  # LERC
            # limit to 4 threads
            return min(numchunks, 4)
        return min(numchunks, TIFF.MAXWORKERS)

    def __enter__(self) -> TiffWriter:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<tifffile.TiffWriter {snipstr(self.filehandle.name, 32)!r}>'