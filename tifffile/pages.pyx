from libc.stdint cimport int32_t, int64_t
from .types cimport DATATYPE, COMPRESSION, PHOTOMETRIC, SAMPLEFORMAT, PREDICTOR, EXTRASAMPLE

cdef class TiffPage:
    """TIFF image file directory (IFD).

    TiffPage instances are not thread-safe. All attributes are read-only.

    Parameters:
        parent:
            TiffFile instance to read page from.
            The file handle position must be at an offset to an IFD structure.
        index:
            Index of page in IFD tree.
        keyframe:
            Not used.

    Raises:
        TiffFileError: Invalid TIFF structure.

    """
    cdef TiffTags tags
    """Tags belonging to page."""
    cdef TiffFile parent
    """TiffFile instance page belongs to."""
    cdef int64_t offset
    """Position of page in file."""
    cdef tuple shape#: tuple[int, ...]
    """Shape of image array in page."""
    cdef object dtype#: numpy.dtype[Any] | None
    """Data type of image array in page."""
    cdef int64_t[5] shaped
    """Normalized 5-dimensional shape of image array in page:

        0. separate samplesperpixel or 1.
        1. imagedepth or 1.
        2. imagelength.
        3. imagewidth.
        4. contig samplesperpixel or 1.

    """
    cdef str axes
    """Character codes for dimensions in image array:
    'S' sample, 'X' width, 'Y' length, 'Z' depth.
    """
    cdef tuple dataoffsets#: tuple[int, ...]
    """Positions of strips or tiles in file."""
    cdef tuple databytecounts#: tuple[int, ...]
    """Size of strips or tiles in file."""
    cdef object _dtype#: numpy.dtype[Any] | None
    cdef tuple _index#: tuple[int, ...]  # index of page in IFD tree

    # default properties; might be updated from tags
    cdef public int subfiletype
    """:py:class:`FILETYPE` kind of image."""
    cdef public int imagewidth
    """Number of columns (pixels per row) in image."""
    cdef public int imagelength
    """Number of rows in image."""
    cdef public int imagedepth
    """Number of Z slices in image."""
    cdef public int tilewidth
    """Number of columns in each tile."""
    cdef public int tilelength
    """Number of rows in each tile."""
    cdef public int tiledepth
    """Number of Z slices in each tile."""
    cdef public int samplesperpixel
    """Number of components per pixel."""
    cdef public int bitspersample
    """Number of bits per pixel component."""
    cdef public int sampleformat
    """:py:class:`SAMPLEFORMAT` type of pixel components."""
    cdef public int rowsperstrip
    """Number of rows per strip."""
    cdef public int compression
    """:py:class:`COMPRESSION` scheme used on image data."""
    cdef public int planarconfig
    """:py:class:`PLANARCONFIG` type of storage of components in pixel."""
    cdef public int fillorder
    """Logical order of bits within byte of image data."""
    cdef public int photometric
    """:py:class:`PHOTOMETRIC` color space of image."""
    cdef public int predictor
    """:py:class:`PREDICTOR` applied to image data before compression."""
    cdef public tuple extrasamples # tuple[int, ...]
    """:py:class:`EXTRASAMPLE` interpretation of extra components in pixel."""
    cdef public int64_t[2] subsampling
    """Subsampling factors used for chrominance components."""
    cdef public tuple subifds # tuple[int, ...]
    """Positions of SubIFDs in file."""
    cdef public bytes jpegtables
    """JPEG quantization and Huffman tables."""
    cdef public bytes jpegheader
    """JPEG header for NDPI."""
    cdef public str software
    """Software used to create image."""
    cdef public str description
    """Subject of image."""
    cdef public str description1
    """Value of second ImageDescription tag."""
    cdef public float nodata # int | float
    """Value used for missing data. The value of the GDAL_NODATA tag or 0."""

    def __cinit__(self):
        self.subfiletype = 0
        self.imagewidth = 0
        self.imagelength = 0
        self.imagedepth = 1
        self.tilewidth = 0
        self.tilelength = 0
        self.tiledepth = 1
        self.samplesperpixel = 1
        self.bitspersample = 1
        self.sampleformat = 1
        self.rowsperstrip = 2**32 - 1
        self.compression = 1
        self.planarconfig = 1
        self.fillorder = 1
        self.photometric = 0
        self.predictor = 1
        self.extrasamples = ()
        self.subsampling = None
        self.subifds = None
        self.jpegtables = None
        self.jpegheader = None
        self.software = ''
        self.description = ''
        self.description1 = ''
        self.nodata = 0

    def __init__(
        self,
        parent: TiffFile,
        /,
        index: int | Sequence[int],
        *,
        keyframe: TiffPage | None = None,
    ) -> None:
        tag: TiffTag | None
        tiff = parent.tiff

        self.parent = parent
        self.shape = ()
        self.shaped = (0, 0, 0, 0, 0)
        self.dtype = self._dtype = None
        self.axes = ''
        self.tags = tags = TiffTags()
        self.dataoffsets = ()
        self.databytecounts = ()
        if isinstance(index, int):
            self._index = (index,)
        else:
            self._index = tuple(index)

        # read IFD structure and its tags from file
        fh = parent.filehandle
        self.offset = fh.tell()  # offset to this IFD
        try:
            tagno: int = struct.unpack(
                tiff.tagnoformat, fh.read(tiff.tagnosize)
            )[0]
            if tagno > 4096:
                raise ValueError(f'suspicious number of tags {tagno}')
        except Exception as exc:
            raise TiffFileError(f'corrupted tag list @{self.offset}') from exc

        tagoffset = self.offset + tiff.tagnosize  # fh.tell()
        tagsize = tagsize_ = tiff.tagsize

        data = fh.read(tagsize * tagno)
        if len(data) != tagsize * tagno:
            raise TiffFileError('corrupted IFD structure')
        if tiff.is_ndpi:
            # patch offsets/values for 64-bit NDPI file
            tagsize = 16
            fh.seek(8, os.SEEK_CUR)
            ext = fh.read(4 * tagno)  # high bits
            data = b''.join(
                data[i * 12 : i * 12 + 12] + ext[i * 4 : i * 4 + 4]
                for i in range(tagno)
            )

        tagindex = -tagsize
        for i in range(tagno):
            tagindex += tagsize
            tagdata = data[tagindex : tagindex + tagsize]
            try:
                tag = TiffTag.fromfile(
                    parent, offset=tagoffset + i * tagsize_, header=tagdata
                )
            except TiffFileError as exc:
                logger().error(f'<TiffTag.fromfile> raised {exc!r:.128}')
                continue
            tags.add(tag)

        if not tags:
            return  # found in FIBICS

        for code, name in TIFF.TAG_ATTRIBUTES.items():
            value = tags.valueof(code)
            if value is None:
                continue
            if code in {270, 305} and not isinstance(value, str):
                # wrong string type for software or description
                continue
            setattr(self, name, value)

        value = tags.valueof(270, index=1)
        if isinstance(value, str):
            self.description1 = value

        if self.subfiletype == 0:
            value = tags.valueof(255)  # SubfileType
            if value == 2:
                self.subfiletype = 0b1  # reduced image
            elif value == 3:
                self.subfiletype = 0b10  # multi-page
        elif not isinstance(self.subfiletype, int):
            # files created by IDEAS
            logger().warning(f'{self!r} invalid {self.subfiletype=}')
            self.subfiletype = 0

        # consolidate private tags; remove them from self.tags
        # if self.is_andor:
        #     self.andor_tags
        # elif self.is_epics:
        #     self.epics_tags
        # elif self.is_ndpi:
        #     self.ndpi_tags
        # if self.is_sis and 34853 in tags:
        #     # TODO: cannot change tag.name
        #     tags[34853].name = 'OlympusSIS2'

        # dataoffsets and databytecounts
        # TileOffsets
        self.dataoffsets = tags.valueof(324)
        if self.dataoffsets is None:
            # StripOffsets
            self.dataoffsets = tags.valueof(273)
            if self.dataoffsets is None:
                # JPEGInterchangeFormat et al.
                self.dataoffsets = tags.valueof(513)
                if self.dataoffsets is None:
                    self.dataoffsets = ()
                    logger().error(f'{self!r} missing data offset tag')
        # TileByteCounts
        self.databytecounts = tags.valueof(325)
        if self.databytecounts is None:
            # StripByteCounts
            self.databytecounts = tags.valueof(279)
            if self.databytecounts is None:
                # JPEGInterchangeFormatLength et al.
                self.databytecounts = tags.valueof(514)

        if (
            self.imagewidth == 0
            and self.imagelength == 0
            and self.dataoffsets
            and self.databytecounts
        ):
            # dimensions may be missing in some RAW formats
            # read dimensions from assumed JPEG encoded segment
            try:
                fh.seek(self.dataoffsets[0])
                (
                    precision,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                ) = jpeg_shape(fh.read(min(self.databytecounts[0], 4096)))
            except Exception:
                pass
            else:
                self.imagelength = imagelength
                self.imagewidth = imagewidth
                self.samplesperpixel = samplesperpixel
                if 258 not in tags:
                    self.bitspersample = 8 if precision <= 8 else 16
                if 262 not in tags and samplesperpixel == 3:
                    self.photometric = PHOTOMETRIC.YCBCR
                if 259 not in tags:
                    self.compression = COMPRESSION.OJPEG
                if 278 not in tags:
                    self.rowsperstrip = imagelength

        elif self.compression == 6:
            # OJPEG hack. See libtiff v4.2.0 tif_dirread.c#L4082
            if 262 not in tags:
                # PhotometricInterpretation missing
                self.photometric = PHOTOMETRIC.YCBCR
            elif self.photometric == 2:
                # RGB -> YCbCr
                self.photometric = PHOTOMETRIC.YCBCR
            if 258 not in tags:
                # BitsPerSample missing
                self.bitspersample = 8
            if 277 not in tags:
                # SamplesPerPixel missing
                if self.photometric in {2, 6}:
                    self.samplesperpixel = 3
                elif self.photometric in {0, 1}:
                    self.samplesperpixel = 3

        elif self.is_lsm or (self.index != 0 and self.parent.is_lsm):
            # correct non standard LSM bitspersample tags
            tags[258]._fix_lsm_bitspersample()
            if self.compression == 1 and self.predictor != 1:
                # work around bug in LSM510 software
                self.predictor = PREDICTOR.NONE

        elif self.is_vista or (self.index != 0 and self.parent.is_vista):
            # ISS Vista writes wrong ImageDepth tag
            self.imagedepth = 1

        elif self.is_stk:
            # read UIC1tag again now that plane count is known
            tag = tags.get(33628)  # UIC1tag
            assert tag is not None
            fh.seek(tag.valueoffset)
            uic2tag = tags.get(33629)  # UIC2tag
            try:
                tag.value = read_uic1tag(
                    fh,
                    tiff.byteorder,
                    tag.dtype,
                    tag.count,
                    0,
                    planecount=uic2tag.count if uic2tag is not None else 1,
                )
            except Exception as exc:
                logger().warning(
                    f'{self!r} <tifffile.read_uic1tag> raised {exc!r:.128}'
                )

        tag = tags.get(50839)
        if tag is not None:
            # decode IJMetadata tag
            try:
                tag.value = imagej_metadata(
                    tag.value,
                    tags[50838].value,  # IJMetadataByteCounts
                    tiff.byteorder,
                )
            except Exception as exc:
                logger().warning(
                    f'{self!r} <tifffile.imagej_metadata> raised {exc!r:.128}'
                )

        # BitsPerSample
        value = tags.valueof(258)
        if value is not None:
            if self.bitspersample != 1:
                pass  # bitspersample was set by ojpeg hack
            elif tags[258].count == 1:
                self.bitspersample = int(value)
            else:
                # LSM might list more items than samplesperpixel
                value = value[: self.samplesperpixel]
                if any(v - value[0] for v in value):
                    self.bitspersample = value
                else:
                    self.bitspersample = int(value[0])

        # SampleFormat
        value = tags.valueof(339)
        if value is not None:
            if tags[339].count == 1:
                try:
                    self.sampleformat = SAMPLEFORMAT(value)
                except ValueError:
                    self.sampleformat = int(value)
            else:
                value = value[: self.samplesperpixel]
                if any(v - value[0] for v in value):
                    try:
                        self.sampleformat = SAMPLEFORMAT(value)
                    except ValueError:
                        self.sampleformat = int(value)
                else:
                    try:
                        self.sampleformat = SAMPLEFORMAT(value[0])
                    except ValueError:
                        self.sampleformat = int(value[0])
        elif self.bitspersample == 32 and (
            self.is_indica or (self.index != 0 and self.parent.is_indica)
        ):
            # IndicaLabsImageWriter does not write SampleFormat tag
            self.sampleformat = SAMPLEFORMAT.IEEEFP

        if 322 in tags:  # TileWidth
            self.rowsperstrip = 0
        elif 257 in tags:  # ImageLength
            if 278 not in tags or tags[278].count > 1:  # RowsPerStrip
                self.rowsperstrip = self.imagelength
            self.rowsperstrip = min(self.rowsperstrip, self.imagelength)
            # self.stripsperimage = int(math.floor(
            #    float(self.imagelength + self.rowsperstrip - 1) /
            #    self.rowsperstrip))

        # determine dtype
        dtypestr = TIFF.SAMPLE_DTYPES.get(
            (self.sampleformat, self.bitspersample), None
        )
        if dtypestr is not None:
            dtype = numpy.dtype(dtypestr)
        else:
            dtype = None
        self.dtype = self._dtype = dtype

        # determine shape of data
        imagelength = self.imagelength
        imagewidth = self.imagewidth
        imagedepth = self.imagedepth
        samplesperpixel = self.samplesperpixel

        if self.photometric == 2 or samplesperpixel > 1:  # PHOTOMETRIC.RGB
            if self.planarconfig == 1:
                self.shaped = (
                    1,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    samplesperpixel,
                )
                if imagedepth == 1:
                    self.shape = (imagelength, imagewidth, samplesperpixel)
                    self.axes = 'YXS'
                else:
                    self.shape = (
                        imagedepth,
                        imagelength,
                        imagewidth,
                        samplesperpixel,
                    )
                    self.axes = 'ZYXS'
            else:
                self.shaped = (
                    samplesperpixel,
                    imagedepth,
                    imagelength,
                    imagewidth,
                    1,
                )
                if imagedepth == 1:
                    self.shape = (samplesperpixel, imagelength, imagewidth)
                    self.axes = 'SYX'
                else:
                    self.shape = (
                        samplesperpixel,
                        imagedepth,
                        imagelength,
                        imagewidth,
                    )
                    self.axes = 'SZYX'
        else:
            self.shaped = (1, imagedepth, imagelength, imagewidth, 1)
            if imagedepth == 1:
                self.shape = (imagelength, imagewidth)
                self.axes = 'YX'
            else:
                self.shape = (imagedepth, imagelength, imagewidth)
                self.axes = 'ZYX'

        if not self.databytecounts:
            self.databytecounts = (
                product(self.shape) * (self.bitspersample // 8),
            )
            if self.compression != 1:
                logger().error(f'{self!r} missing ByteCounts tag')

        if imagelength and self.rowsperstrip and not self.is_lsm:
            # fix incorrect number of strip bytecounts and offsets
            maxstrips = (
                int(
                    math.floor(imagelength + self.rowsperstrip - 1)
                    / self.rowsperstrip
                )
                * self.imagedepth
            )
            if self.planarconfig == 2:
                maxstrips *= self.samplesperpixel
            if maxstrips != len(self.databytecounts):
                logger().error(
                    f'{self!r} incorrect StripByteCounts count '
                    f'({len(self.databytecounts)} != {maxstrips})'
                )
                self.databytecounts = self.databytecounts[:maxstrips]
            if maxstrips != len(self.dataoffsets):
                logger().error(
                    f'{self!r} incorrect StripOffsets count '
                    f'({len(self.dataoffsets)} != {maxstrips})'
                )
                self.dataoffsets = self.dataoffsets[:maxstrips]

        value = tags.valueof(42113)  # GDAL_NODATA
        if value is not None and dtype is not None:
            try:
                pytype = type(dtype.type(0).item())
                value = value.replace(',', '.')  # comma decimal separator
                self.nodata = pytype(value)
                if not numpy.can_cast(
                    numpy.min_scalar_type(self.nodata), self.dtype
                ):
                    raise ValueError(
                        f'{self.nodata} is not castable to {self.dtype}'
                    )
            except Exception as exc:
                logger().warning(
                    f'{self!r} parsing GDAL_NODATA tag raised {exc!r:.128}'
                )
                self.nodata = 0

        mcustarts = tags.valueof(65426)
        if mcustarts is not None and self.is_ndpi:
            # use NDPI JPEG McuStarts as tile offsets
            mcustarts = mcustarts.astype(numpy.int64)
            high = tags.valueof(65432)
            if high is not None:
                # McuStartsHighBytes
                high = high.astype(numpy.uint64)
                high <<= 32
                mcustarts += high.astype(numpy.int64)
            fh.seek(self.dataoffsets[0])
            jpegheader = fh.read(mcustarts[0])
            try:
                (
                    self.tilelength,
                    self.tilewidth,
                    self.jpegheader,
                ) = ndpi_jpeg_tile(jpegheader)
            except ValueError as exc:
                logger().warning(
                    f'{self!r} <tifffile.ndpi_jpeg_tile> raised {exc!r:.128}'
                )
            else:
                # TODO: optimize tuple(ndarray.tolist())
                databytecounts = numpy.diff(
                    mcustarts, append=self.databytecounts[0]
                )
                self.databytecounts = tuple(
                    databytecounts.tolist()  # type: ignore[arg-type]
                )
                mcustarts += self.dataoffsets[0]
                self.dataoffsets = tuple(mcustarts.tolist())

    @cached_property
    def decode(
        self,
    ) -> Callable[
        ...,
        tuple[
            NDArray[Any] | None,
            tuple[int, int, int, int, int],
            tuple[int, int, int, int],
        ],
    ]:
        """Return decoded segment, its shape, and indices in image.

        The decode function is implemented as a closure and has the following
        signature:

        Parameters:
            data (Union[bytes, None]):
                Encoded bytes of segment (strip or tile) or None for empty
                segments.
            index (int):
                Index of segment in Offsets and Bytecount tag values.
            jpegtables (Optional[bytes]):
                For JPEG compressed segments only, value of JPEGTables tag
                if any.

        Returns:
            - Decoded segment or None for empty segments.
            - Position of segment in image array of normalized shape
              (separate sample, depth, length, width, contig sample).
            - Shape of segment (depth, length, width, contig samples).
              The shape of strips depends on their linear index.

        Raises:
            ValueError or NotImplementedError:
                Decoding is not supported.
            TiffFileError:
                Invalid TIFF structure.

        """
        if self.hash in self.parent._parent._decoders:
            return self.parent._parent._decoders[self.hash]

        def cache(decode, /):
            self.parent._parent._decoders[self.hash] = decode
            return decode

        if self.dtype is None or self._dtype is None:

            def decode_raise_dtype(*args, **kwargs):
                raise ValueError(
                    'data type not supported '
                    f'(SampleFormat {self.sampleformat}, '
                    f'{self.bitspersample}-bit)'
                )

            return cache(decode_raise_dtype)

        if 0 in self.shaped:

            def decode_raise_empty(*args, **kwargs):
                raise ValueError('empty image')

            return cache(decode_raise_empty)

        try:
            if self.compression == 1:
                decompress = None
            else:
                decompress = TIFF.DECOMPRESSORS[self.compression]
            if (
                self.compression in {65000, 65001, 65002}
                and not self.parent.is_eer
            ):
                raise KeyError(self.compression)
        except KeyError as exc:

            def decode_raise_compression(*args, exc=str(exc)[1:-1], **kwargs):
                raise ValueError(f'{exc}')

            return cache(decode_raise_compression)

        try:
            if self.predictor == 1:
                unpredict = None
            else:
                unpredict = TIFF.UNPREDICTORS[self.predictor]
        except KeyError as exc:
            if self.compression in TIFF.IMAGE_COMPRESSIONS:
                logger().warning(
                    f'{self!r} ignoring predictor {self.predictor}'
                )
                unpredict = None

            else:

                def decode_raise_predictor(
                    *args, exc=str(exc)[1:-1], **kwargs
                ):
                    raise ValueError(f'{exc}')

                return cache(decode_raise_predictor)

        if self.tags.get(339) is not None:
            tag = self.tags[339]  # SampleFormat
            if tag.count != 1 and any(i - tag.value[0] for i in tag.value):

                def decode_raise_sampleformat(*args, **kwargs):
                    raise ValueError(
                        f'sample formats do not match {tag.value}'
                    )

                return cache(decode_raise_sampleformat)

        if self.is_subsampled and (
            self.compression not in {6, 7, 34892, 33007}
            or self.planarconfig == 2
        ):

            def decode_raise_subsampling(*args, **kwargs):
                raise NotImplementedError(
                    'chroma subsampling not supported without JPEG compression'
                )

            return cache(decode_raise_subsampling)

        if self.compression == 50001 and self.samplesperpixel == 4:
            # WebP segments may be missing all-opaque alpha channel
            def decompress_webp_rgba(data, out=None):
                return imagecodecs.webp_decode(data, hasalpha=True, out=out)

            decompress = decompress_webp_rgba

        # normalize segments shape to [depth, length, width, contig]
        if self.is_tiled:
            stshape = (
                self.tiledepth,
                self.tilelength,
                self.tilewidth,
                self.samplesperpixel if self.planarconfig == 1 else 1,
            )
        else:
            stshape = (
                1,
                self.rowsperstrip,
                self.imagewidth,
                self.samplesperpixel if self.planarconfig == 1 else 1,
            )

        stdepth, stlength, stwidth, samples = stshape
        _, imdepth, imlength, imwidth, samples = self.shaped

        if self.is_tiled:
            width = (imwidth + stwidth - 1) // stwidth
            length = (imlength + stlength - 1) // stlength
            depth = (imdepth + stdepth - 1) // stdepth

            def indices(
                segmentindex: int, /
            ) -> tuple[
                tuple[int, int, int, int, int], tuple[int, int, int, int]
            ]:
                # return indices and shape of tile in image array
                return (
                    (
                        segmentindex // (width * length * depth),
                        (segmentindex // (width * length)) % depth * stdepth,
                        (segmentindex // width) % length * stlength,
                        segmentindex % width * stwidth,
                        0,
                    ),
                    stshape,
                )

            def reshape(
                data: NDArray[Any],
                indices: tuple[int, int, int, int, int],
                shape: tuple[int, int, int, int],
                /,
            ) -> NDArray[Any]:
                # return reshaped tile or raise TiffFileError
                size = shape[0] * shape[1] * shape[2] * shape[3]
                if data.ndim == 1 and data.size > size:
                    # decompression / unpacking might return too many bytes
                    data = data[:size]
                if data.size == size:
                    # complete tile
                    # data might be non-contiguous; cannot reshape inplace
                    return data.reshape(shape)
                try:
                    # data fills remaining space
                    # found in JPEG/PNG compressed tiles
                    return data.reshape(
                        (
                            min(imdepth - indices[1], shape[0]),
                            min(imlength - indices[2], shape[1]),
                            min(imwidth - indices[3], shape[2]),
                            samples,
                        )
                    )
                except ValueError:
                    pass
                try:
                    # data fills remaining horizontal space
                    # found in tiled GeoTIFF
                    return data.reshape(
                        (
                            min(imdepth - indices[1], shape[0]),
                            min(imlength - indices[2], shape[1]),
                            shape[2],
                            samples,
                        )
                    )
                except ValueError:
                    pass
                raise TiffFileError(
                    f'corrupted tile @ {indices} cannot be reshaped from '
                    f'{data.shape} to {shape}'
                )

            def pad(
                data: NDArray[Any], shape: tuple[int, int, int, int], /
            ) -> tuple[NDArray[Any], tuple[int, int, int, int]]:
                # pad tile to shape
                if data.shape == shape:
                    return data, shape
                padwidth = [(0, i - j) for i, j in zip(shape, data.shape)]
                data = numpy.pad(data, padwidth, constant_values=self.nodata)
                return data, shape

            def pad_none(
                shape: tuple[int, int, int, int], /
            ) -> tuple[int, int, int, int]:
                # return shape of tile
                return shape

        else:
            # strips
            length = (imlength + stlength - 1) // stlength

            def indices(
                segmentindex: int, /
            ) -> tuple[
                tuple[int, int, int, int, int], tuple[int, int, int, int]
            ]:
                # return indices and shape of strip in image array
                indices = (
                    segmentindex // (length * imdepth),
                    (segmentindex // length) % imdepth * stdepth,
                    segmentindex % length * stlength,
                    0,
                    0,
                )
                shape = (
                    stdepth,
                    min(stlength, imlength - indices[2]),
                    stwidth,
                    samples,
                )
                return indices, shape

            def reshape(
                data: NDArray[Any],
                indices: tuple[int, int, int, int, int],
                shape: tuple[int, int, int, int],
                /,
            ) -> NDArray[Any]:
                # return reshaped strip or raise TiffFileError
                size = shape[0] * shape[1] * shape[2] * shape[3]
                if data.ndim == 1 and data.size > size:
                    # decompression / unpacking might return too many bytes
                    data = data[:size]
                if data.size == size:
                    # expected size
                    try:
                        data.shape = shape
                    except AttributeError:
                        # incompatible shape for in-place modification
                        # decoder returned non-contiguous array
                        data = data.reshape(shape)
                    return data
                datashape = data.shape
                try:
                    # too many rows?
                    data.shape = shape[0], -1, shape[2], shape[3]
                    data = data[:, : shape[1]]
                    data.shape = shape
                    return data
                except ValueError:
                    pass
                raise TiffFileError(
                    'corrupted strip cannot be reshaped from '
                    f'{datashape} to {shape}'
                )

            def pad(
                data: NDArray[Any], shape: tuple[int, int, int, int], /
            ) -> tuple[NDArray[Any], tuple[int, int, int, int]]:
                # pad strip length to rowsperstrip
                shape = (shape[0], stlength, shape[2], shape[3])
                if data.shape == shape:
                    return data, shape
                padwidth = [
                    (0, 0),
                    (0, stlength - data.shape[1]),
                    (0, 0),
                    (0, 0),
                ]
                data = numpy.pad(data, padwidth, constant_values=self.nodata)
                return data, shape

            def pad_none(
                shape: tuple[int, int, int, int], /
            ) -> tuple[int, int, int, int]:
                # return shape of strip
                return (shape[0], stlength, shape[2], shape[3])

        if self.compression in {6, 7, 34892, 33007}:
            # JPEG needs special handling
            if self.fillorder == 2:
                logger().debug(f'{self!r} disabling LSB2MSB for JPEG')
            if unpredict:
                logger().debug(f'{self!r} disabling predictor for JPEG')
            if 28672 in self.tags:  # SonyRawFileType
                logger().warning(
                    f'{self!r} SonyRawFileType might need additional '
                    'unpacking (see issue #95)'
                )

            colorspace, outcolorspace = jpeg_decode_colorspace(
                self.photometric,
                self.planarconfig,
                self.extrasamples,
                self.is_jfif,
            )

            def decode_jpeg(
                data: bytes | None,
                index: int,
                /,
                *,
                jpegtables: bytes | None = None,
                jpegheader: bytes | None = None,
                _fullsize: bool = False,
            ) -> tuple[
                NDArray[Any] | None,
                tuple[int, int, int, int, int],
                tuple[int, int, int, int],
            ]:
                # return decoded segment, its shape, and indices in image
                segmentindex, shape = indices(index)
                if data is None:
                    if _fullsize:
                        shape = pad_none(shape)
                    return data, segmentindex, shape
                data_array: NDArray[Any] = imagecodecs.jpeg_decode(
                    data,
                    bitspersample=self.bitspersample,
                    tables=jpegtables,
                    header=jpegheader,
                    colorspace=colorspace,
                    outcolorspace=outcolorspace,
                    shape=shape[1:3],
                )
                data_array = reshape(data_array, segmentindex, shape)
                if _fullsize:
                    data_array, shape = pad(data_array, shape)
                return data_array, segmentindex, shape

            return cache(decode_jpeg)

        if self.compression in {65000, 65001, 65002}:
            # EER decoder requires shape and extra args

            if self.compression == 65002:
                rlebits = int(self.tags.valueof(65007, 7))
                horzbits = int(self.tags.valueof(65008, 2))
                vertbits = int(self.tags.valueof(65009, 2))
            elif self.compression == 65001:
                rlebits = 7
                horzbits = 2
                vertbits = 2
            else:
                rlebits = 8
                horzbits = 2
                vertbits = 2

            def decode_eer(
                data: bytes | None,
                index: int,
                /,
                *,
                jpegtables: bytes | None = None,
                jpegheader: bytes | None = None,
                _fullsize: bool = False,
            ) -> tuple[
                NDArray[Any] | None,
                tuple[int, int, int, int, int],
                tuple[int, int, int, int],
            ]:
                # return decoded eer segment, its shape, and indices in image
                segmentindex, shape = indices(index)
                if data is None:
                    if _fullsize:
                        shape = pad_none(shape)
                    return data, segmentindex, shape
                data_array = decompress(
                    data,
                    shape=shape[1:3],
                    rlebits=rlebits,
                    horzbits=horzbits,
                    vertbits=vertbits,
                    superres=False,
                )  # type: ignore[call-arg, misc]
                return data_array.reshape(shape), segmentindex, shape

            return cache(decode_eer)

        if self.compression == 48124:
            # Jetraw requires pre-allocated output buffer

            def decode_jetraw(
                data: bytes | None,
                index: int,
                /,
                *,
                jpegtables: bytes | None = None,
                jpegheader: bytes | None = None,
                _fullsize: bool = False,
            ) -> tuple[
                NDArray[Any] | None,
                tuple[int, int, int, int, int],
                tuple[int, int, int, int],
            ]:
                # return decoded segment, its shape, and indices in image
                segmentindex, shape = indices(index)
                if data is None:
                    if _fullsize:
                        shape = pad_none(shape)
                    return data, segmentindex, shape
                data_array = numpy.zeros(shape, numpy.uint16)
                decompress(data, out=data_array)  # type: ignore[misc]
                return data_array.reshape(shape), segmentindex, shape

            return cache(decode_jetraw)

        if self.compression in TIFF.IMAGE_COMPRESSIONS:
            # presume codecs always return correct dtype, native byte order...
            if self.fillorder == 2:
                logger().debug(
                    f'{self!r} '
                    f'disabling LSB2MSB for compression {self.compression}'
                )
            if unpredict:
                logger().debug(
                    f'{self!r} '
                    f'disabling predictor for compression {self.compression}'
                )

            def decode_image(
                data: bytes | None,
                index: int,
                /,
                *,
                jpegtables: bytes | None = None,
                jpegheader: bytes | None = None,
                _fullsize: bool = False,
            ) -> tuple[
                NDArray[Any] | None,
                tuple[int, int, int, int, int],
                tuple[int, int, int, int],
            ]:
                # return decoded segment, its shape, and indices in image
                segmentindex, shape = indices(index)
                if data is None:
                    if _fullsize:
                        shape = pad_none(shape)
                    return data, segmentindex, shape
                data_array: NDArray[Any]
                data_array = decompress(data)  # type: ignore[misc]
                # del data
                data_array = reshape(data_array, segmentindex, shape)
                if _fullsize:
                    data_array, shape = pad(data_array, shape)
                return data_array, segmentindex, shape

            return cache(decode_image)

        dtype = numpy.dtype(self.parent.byteorder + self._dtype.char)

        if self.sampleformat == 5:
            # complex integer
            if unpredict is not None:
                raise NotImplementedError(
                    'unpredicting complex integers not supported'
                )

            itype = numpy.dtype(
                f'{self.parent.byteorder}i{self.bitspersample // 16}'
            )
            ftype = numpy.dtype(
                f'{self.parent.byteorder}f{dtype.itemsize // 2}'
            )

            def unpack(data: bytes, /) -> NDArray[Any]:
                # return complex integer as numpy.complex
                return numpy.frombuffer(data, itype).astype(ftype).view(dtype)

        elif self.bitspersample in {8, 16, 32, 64, 128}:
            # regular data types

            if (self.bitspersample * stwidth * samples) % 8:
                raise ValueError('data and sample size mismatch')
            if self.predictor in {3, 34894, 34895}:  # PREDICTOR.FLOATINGPOINT
                # floating-point horizontal differencing decoder needs
                # raw byte order
                dtype = numpy.dtype(self._dtype.char)

            def unpack(data: bytes, /) -> NDArray[Any]:
                # return numpy array from buffer
                try:
                    # read only numpy array
                    return numpy.frombuffer(data, dtype)
                except ValueError:
                    # for example, LZW strips may be missing EOI
                    bps = self.bitspersample // 8
                    size = (len(data) // bps) * bps
                    return numpy.frombuffer(data[:size], dtype)

        elif isinstance(self.bitspersample, tuple):
            # for example, RGB 565
            def unpack(data: bytes, /) -> NDArray[Any]:
                # return numpy array from packed integers
                return unpack_rgb(data, dtype, self.bitspersample)

        elif self.bitspersample == 24 and dtype.char == 'f':
            # float24
            if unpredict is not None:
                # floatpred_decode requires numpy.float24, which does not exist
                raise NotImplementedError('unpredicting float24 not supported')

            def unpack(data: bytes, /) -> NDArray[Any]:
                # return numpy.float32 array from float24
                return imagecodecs.float24_decode(
                    data, byteorder=self.parent.byteorder
                )

        else:
            # bilevel and packed integers
            def unpack(data: bytes, /) -> NDArray[Any]:
                # return NumPy array from packed integers
                return imagecodecs.packints_decode(
                    data, dtype, self.bitspersample, runlen=stwidth * samples
                )

        def decode_other(
            data: bytes | None,
            index: int,
            /,
            *,
            jpegtables: bytes | None = None,
            jpegheader: bytes | None = None,
            _fullsize: bool = False,
        ) -> tuple[
            NDArray[Any] | None,
            tuple[int, int, int, int, int],
            tuple[int, int, int, int],
        ]:
            # return decoded segment, its shape, and indices in image
            segmentindex, shape = indices(index)
            if data is None:
                if _fullsize:
                    shape = pad_none(shape)
                return data, segmentindex, shape
            if self.fillorder == 2:
                data = imagecodecs.bitorder_decode(data)
            if decompress is not None:
                # TODO: calculate correct size for packed integers
                size = shape[0] * shape[1] * shape[2] * shape[3]
                data = decompress(data, out=size * dtype.itemsize)
            data_array = unpack(data)  # type: ignore[arg-type]
            # del data
            data_array = reshape(data_array, segmentindex, shape)
            data_array = data_array.astype('=' + dtype.char, copy=False)
            if unpredict is not None:
                # unpredict is faster with native byte order
                data_array = unpredict(data_array, axis=-2, out=data_array)
            if _fullsize:
                data_array, shape = pad(data_array, shape)
            return data_array, segmentindex, shape

        return cache(decode_other)

    def segments(
        self,
        *,
        lock: threading.RLock | NullContext | None = None,
        maxworkers: int | None = None,
        func: Callable[..., Any] | None = None,  # TODO: type this
        sort: bool = False,
        buffersize: int | None = None,
        _fullsize: bool | None = None,
    ) -> Iterator[
        tuple[
            NDArray[Any] | None,
            tuple[int, int, int, int, int],
            tuple[int, int, int, int],
        ]
    ]:
        """Return iterator over decoded tiles or strips.

        Parameters:
            lock:
                Reentrant lock to synchronize file seeks and reads.
            maxworkers:
                Maximum number of threads to concurrently decode segments.
            func:
                Function to process decoded segment.
            sort:
                Read segments from file in order of their offsets.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.
            _fullsize:
                Internal use.

        Yields:
            - Decoded segment or None for empty segments.
            - Position of segment in image array of normalized shape
              (separate sample, depth, length, width, contig sample).
            - Shape of segment (depth, length, width, contig samples).
              The shape of strips depends on their linear index.

        """
        keyframe = self.keyframe  # self or keyframe
        fh = self.parent.filehandle
        if lock is None:
            lock = fh.lock
        if _fullsize is None:
            _fullsize = keyframe.is_tiled

        decodeargs: dict[str, Any] = {'_fullsize': bool(_fullsize)}
        if keyframe.compression in {6, 7, 34892, 33007}:  # JPEG
            decodeargs['jpegtables'] = self.jpegtables
            decodeargs['jpegheader'] = keyframe.jpegheader

        if func is None:

            def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
                return decode(*args, **decodeargs)

        else:

            def decode(args, decodeargs=decodeargs, decode=keyframe.decode):
                return func(decode(*args, **decodeargs))

        if maxworkers is None or maxworkers < 1:
            maxworkers = keyframe.maxworkers
        if maxworkers < 2:
            for segment in fh.read_segments(
                self.dataoffsets,
                self.databytecounts,
                lock=lock,
                sort=sort,
                buffersize=buffersize,
                flat=True,
            ):
                yield decode(segment)
        else:
            # reduce memory overhead by processing chunks of up to
            # buffersize of segments because ThreadPoolExecutor.map is not
            # collecting iterables lazily
            with ThreadPoolExecutor(maxworkers) as executor:
                for segments in fh.read_segments(
                    self.dataoffsets,
                    self.databytecounts,
                    lock=lock,
                    sort=sort,
                    buffersize=buffersize,
                    flat=False,
                ):
                    yield from executor.map(decode, segments)

    def asarray(
        self,
        *,
        out: OutputType = None,
        squeeze: bool = True,
        lock: threading.RLock | NullContext | None = None,
        maxworkers: int | None = None,
        buffersize: int | None = None,
    ) -> NDArray[Any]:
        """Return image from page as NumPy array.

        Parameters:
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
            squeeze:
                Remove all length-1 dimensions (except X and Y) from
                image array.
                If *False*, return the image array with normalized
                5-dimensional shape :py:attr:`TiffPage.shaped`.
            lock:
                Reentrant lock to synchronize seeks and reads from file.
                The default is the lock of the parent's file handle.
            maxworkers:
                Maximum number of threads to concurrently decode segments.
                If *None* or *0*, use up to :py:attr:`_TIFF.MAXWORKERS`
                threads. See remarks in :py:meth:`TiffFile.asarray`.
            buffersize:
                Approximate number of bytes to read from file in one pass.
                The default is :py:attr:`_TIFF.BUFFERSIZE`.

        Returns:
            NumPy array of decompressed, unpredicted, and unpacked image data
            read from Strip/Tile Offsets/ByteCounts, formatted according to
            shape and dtype metadata found in tags and arguments.
            Photometric conversion, premultiplied alpha, orientation, and
            colorimetry corrections are not applied.
            Specifically, CMYK images are not converted to RGB, MinIsWhite
            images are not inverted, color palettes are not applied,
            gamma is not corrected, and CFA images are not demosaciced.
            Exception are YCbCr JPEG compressed images, which are converted to
            RGB.

        Raises:
            ValueError:
                Format of image in file is not supported and cannot be decoded.

        """
        keyframe = self.keyframe  # self or keyframe

        if 0 in keyframe.shaped or keyframe._dtype is None:
            return numpy.empty((0,), keyframe.dtype)

        if len(self.dataoffsets) == 0:
            raise TiffFileError('missing data offset')

        fh = self.parent.filehandle
        if lock is None:
            lock = fh.lock

        if (
            isinstance(out, str)
            and out == 'memmap'
            and keyframe.is_memmappable
        ):
            # direct memory map array in file
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file', UserWarning
                    )
                    fh.open()
                result = fh.memmap_array(
                    keyframe.parent.byteorder + keyframe._dtype.char,
                    keyframe.shaped,
                    offset=self.dataoffsets[0],
                )

        elif keyframe.is_contiguous:
            # read contiguous bytes to array
            if keyframe.is_subsampled:
                raise NotImplementedError('chroma subsampling not supported')
            if out is not None:
                out = create_output(out, keyframe.shaped, keyframe._dtype)
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file', UserWarning
                    )
                    fh.open()
                fh.seek(self.dataoffsets[0])
                result = fh.read_array(
                    keyframe.parent.byteorder + keyframe._dtype.char,
                    product(keyframe.shaped),
                    out=out,
                )
            if keyframe.fillorder == 2:
                result = imagecodecs.bitorder_decode(result, out=result)
            if keyframe.predictor != 1:
                # predictors without compression
                unpredict = TIFF.UNPREDICTORS[keyframe.predictor]
                if keyframe.predictor == 1:
                    result = unpredict(result, axis=-2, out=result)
                else:
                    # floatpred cannot decode in-place
                    out = unpredict(result, axis=-2, out=result)
                    result[:] = out

        elif (
            keyframe.jpegheader is not None
            and keyframe is self
            and 273 in self.tags  # striped ...
            and self.is_tiled  # but reported as tiled
            # TODO: imagecodecs can decode larger JPEG
            and self.imagewidth <= 65500
            and self.imagelength <= 65500
        ):
            # decode the whole NDPI JPEG strip
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file', UserWarning
                    )
                    fh.open()
                fh.seek(self.tags[273].value[0])  # StripOffsets
                data = fh.read(self.tags[279].value[0])  # StripByteCounts
            decompress = TIFF.DECOMPRESSORS[self.compression]
            result = decompress(
                data,
                bitspersample=self.bitspersample,
                out=out,
                # shape=(self.imagelength, self.imagewidth)
            )
            del data

        else:
            # decode individual strips or tiles
            with lock:
                closed = fh.closed
                if closed:
                    warnings.warn(
                        f'{self!r} reading array from closed file', UserWarning
                    )
                    fh.open()
                keyframe.decode  # init TiffPage.decode function under lock

            result = create_output(out, keyframe.shaped, keyframe._dtype)

            def func(
                decoderesult: tuple[
                    NDArray[Any] | None,
                    tuple[int, int, int, int, int],
                    tuple[int, int, int, int],
                ],
                keyframe: TiffPage = keyframe,
                out: NDArray[Any] = result,
            ) -> None:
                # copy decoded segments to output array
                segment, (s, d, h, w, _), shape = decoderesult
                if segment is None:
                    out[
                        s, d : d + shape[0], h : h + shape[1], w : w + shape[2]
                    ] = keyframe.nodata
                else:
                    out[
                        s, d : d + shape[0], h : h + shape[1], w : w + shape[2]
                    ] = segment[
                        : keyframe.imagedepth - d,
                        : keyframe.imagelength - h,
                        : keyframe.imagewidth - w,
                    ]
                # except IndexError:
                #     pass  # corrupted file, for example, with too many strips

            for _ in self.segments(
                func=func,
                lock=lock,
                maxworkers=maxworkers,
                buffersize=buffersize,
                sort=True,
                _fullsize=False,
            ):
                pass

        result.shape = keyframe.shaped
        if squeeze:
            try:
                result.shape = keyframe.shape
            except ValueError as exc:
                logger().warning(
                    f'{self!r} <asarray> failed to reshape '
                    f'{result.shape} to {keyframe.shape}, raised {exc!r:.128}'
                )

        if closed:
            # TODO: close file if an exception occurred above
            fh.close()
        return result

    def aszarr(self, **kwargs: Any) -> ZarrTiffStore:
        """Return image from page as Zarr 2 store.

        Parameters:
            **kwarg: Passed to :py:class:`ZarrTiffStore`.

        """
        return ZarrTiffStore(self, **kwargs)

    def asrgb(
        self,
        *,
        uint8: bool = False,
        alpha: Container[int] | None = None,
        **kwargs: Any,
    ) -> NDArray[Any]:
        """Return image as RGB(A). Work in progress. Do not use.

        :meta private:

        """
        data = self.asarray(**kwargs)
        keyframe = self.keyframe  # self or keyframe

        if keyframe.photometric == PHOTOMETRIC.PALETTE:
            colormap = keyframe.colormap
            if colormap is None:
                raise ValueError('no colormap')
            if (
                colormap.shape[1] < 2**keyframe.bitspersample
                or keyframe.dtype is None
                or keyframe.dtype.char not in 'BH'
            ):
                raise ValueError('cannot apply colormap')
            if uint8:
                if colormap.max() > 255:
                    colormap >>= 8
                colormap = colormap.astype(numpy.uint8)
            if 'S' in keyframe.axes:
                data = data[..., 0] if keyframe.planarconfig == 1 else data[0]
            data = apply_colormap(data, colormap)

        elif keyframe.photometric == PHOTOMETRIC.RGB:
            if keyframe.extrasamples:
                if alpha is None:
                    alpha = EXTRASAMPLE
                for i, exs in enumerate(keyframe.extrasamples):
                    if exs in EXTRASAMPLE:
                        if keyframe.planarconfig == 1:
                            data = data[..., [0, 1, 2, 3 + i]]
                        else:
                            data = data[:, [0, 1, 2, 3 + i]]
                        break
            else:
                if keyframe.planarconfig == 1:
                    data = data[..., :3]
                else:
                    data = data[:, :3]
            # TODO: convert to uint8?

        elif keyframe.photometric == PHOTOMETRIC.MINISBLACK:
            raise NotImplementedError
        elif keyframe.photometric == PHOTOMETRIC.MINISWHITE:
            raise NotImplementedError
        elif keyframe.photometric == PHOTOMETRIC.SEPARATED:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return data

    def _gettags(
        self,
        codes: Container[int] | None = None,
        /,
        lock: threading.RLock | None = None,
    ) -> list[tuple[int, TiffTag]]:
        """Return list of (code, TiffTag)."""
        return [
            (tag.code, tag)
            for tag in self.tags
            if codes is None or tag.code in codes
        ]

    def _nextifd(self) -> int:
        """Return offset to next IFD from file."""
        fh = self.parent.filehandle
        tiff = self.parent.tiff
        fh.seek(self.offset)
        tagno = struct.unpack(tiff.tagnoformat, fh.read(tiff.tagnosize))[0]
        fh.seek(self.offset + tiff.tagnosize + tagno * tiff.tagsize)
        return int(
            struct.unpack(tiff.offsetformat, fh.read(tiff.offsetsize))[0]
        )

    def aspage(self) -> TiffPage:
        """Return TiffPage instance."""
        return self

    @property
    def index(self) -> int:
        """Index of page in IFD chain."""
        return self._index[-1]

    @property
    def treeindex(self) -> tuple[int, ...]:
        """Index of page in IFD tree."""
        return self._index

    @property
    def keyframe(self) -> TiffPage:
        """Self."""
        return self

    @keyframe.setter
    def keyframe(self, index: TiffPage) -> None:
        return

    @property
    def name(self) -> str:
        """Name of image array."""
        index = self._index if len(self._index) > 1 else self._index[0]
        return f'TiffPage {index}'

    @property
    def ndim(self) -> int:
        """Number of dimensions in image array."""
        return len(self.shape)

    @cached_property
    def dims(self) -> tuple[str, ...]:
        """Names of dimensions in image array."""
        names = TIFF.AXES_NAMES
        return tuple(names[ax] for ax in self.axes)

    @cached_property
    def sizes(self) -> dict[str, int]:
        """Ordered map of dimension names to lengths."""
        shape = self.shape
        names = TIFF.AXES_NAMES
        return {names[ax]: shape[i] for i, ax in enumerate(self.axes)}

    @cached_property
    def coords(self) -> dict[str, NDArray[Any]]:
        """Ordered map of dimension names to coordinate arrays."""
        resolution = self.get_resolution()
        coords: dict[str, NDArray[Any]] = {}

        for ax, size in zip(self.axes, self.shape):
            name = TIFF.AXES_NAMES[ax]
            value = None
            step: int | float = 1

            if ax == 'X':
                step = resolution[0]
            elif ax == 'Y':
                step = resolution[1]
            elif ax == 'S':
                value = self._sample_names()
            elif ax == 'Z':
                # a ZResolution tag doesn't exist.
                # use XResolution if it agrees with YResolution
                if resolution[0] == resolution[1]:
                    step = resolution[0]

            if value is not None:
                coords[name] = numpy.asarray(value)
            elif step == 0 or step == 1 or size == 0:
                coords[name] = numpy.arange(size)
            else:
                coords[name] = numpy.linspace(
                    0, size / step, size, endpoint=False, dtype=numpy.float32
                )
            assert len(coords[name]) == size
        return coords

    @cached_property
    def attr(self) -> dict[str, Any]:
        """Arbitrary metadata associated with image array."""
        # TODO: what to return?
        return {}

    @cached_property
    def size(self) -> int:
        """Number of elements in image array."""
        return product(self.shape)

    @cached_property
    def nbytes(self) -> int:
        """Number of bytes in image array."""
        if self.dtype is None:
            return 0
        return self.size * self.dtype.itemsize

    @property
    def colormap(self) -> NDArray[numpy.uint16] | None:
        """Value of Colormap tag."""
        return self.tags.valueof(320)

    @property
    def iccprofile(self) -> bytes | None:
        """Value of InterColorProfile tag."""
        return self.tags.valueof(34675)

    @property
    def transferfunction(self) -> NDArray[numpy.uint16] | None:
        """Value of TransferFunction tag."""
        return self.tags.valueof(301)

    def get_resolution(
        self,
        unit: RESUNIT | int | str | None = None,
        scale: float | int | None = None,
    ) -> tuple[int | float, int | float]:
        """Return number of pixels per unit in X and Y dimensions.

        By default, the XResolution and YResolution tag values are returned.
        Missing tag values are set to 1.

        Parameters:
            unit:
                Unit of measurement of returned values.
                The default is the value of the ResolutionUnit tag.
            scale:
                Factor to convert resolution values to meter unit.
                The default is determined from the ResolutionUnit tag.

        """
        scales = {
            1: 1,  # meter, no unit
            2: 100 / 2.54,  # INCH
            3: 100,  # CENTIMETER
            4: 1000,  # MILLIMETER
            5: 1000000,  # MICROMETER
        }
        if unit is not None:
            unit = enumarg(RESUNIT, unit)
            try:
                if scale is None:
                    resolutionunit = self.tags.valueof(296, default=2)
                    scale = scales[resolutionunit]
            except Exception as exc:
                logger().warning(
                    f'{self!r} <get_resolution> raised {exc!r:.128}'
                )
                scale = 1
            else:
                scale2 = scales[unit]
                if scale % scale2 == 0:
                    scale //= scale2
                else:
                    scale /= scale2
        elif scale is None:
            scale = 1

        resolution: list[int | float] = []
        n: int
        d: int
        for code in 282, 283:
            try:
                n, d = self.tags.valueof(code, default=(1, 1))
                if d == 0:
                    value = n * scale
                elif n % d == 0:
                    value = n // d * scale
                else:
                    value = n / d * scale
            except Exception:
                value = 1
            resolution.append(value)
        return resolution[0], resolution[1]

    @cached_property
    def resolution(self) -> tuple[float, float]:
        """Number of pixels per resolutionunit in X and Y directions."""
        # values are returned in (somewhat unexpected) XY order to
        # keep symmetry with the TiffWriter.write resolution argument
        resolution = self.get_resolution()
        return float(resolution[0]), float(resolution[1])

    @property
    def resolutionunit(self) -> int:
        """Unit of measurement for X and Y resolutions."""
        return self.tags.valueof(296, default=2)

    @property
    def datetime(self) -> DateTime | None:
        """Date and time of image creation."""
        value = self.tags.valueof(306)
        if value is None:
            return None
        try:
            return strptime(value)
        except Exception:
            pass
        return None

    @property
    def tile(self) -> tuple[int, ...] | None:
        """Tile depth, length, and width."""
        if not self.is_tiled:
            return None
        if self.tiledepth > 1:
            return (self.tiledepth, self.tilelength, self.tilewidth)
        return (self.tilelength, self.tilewidth)

    @cached_property
    def chunks(self) -> tuple[int, ...]:
        """Shape of images in tiles or strips."""
        shape: list[int] = []
        if self.tiledepth > 1:
            shape.append(self.tiledepth)
        if self.is_tiled:
            shape.extend((self.tilelength, self.tilewidth))
        else:
            shape.extend((self.rowsperstrip, self.imagewidth))
        if self.planarconfig == 1 and self.samplesperpixel > 1:
            shape.append(self.samplesperpixel)
        return tuple(shape)

    @cached_property
    def chunked(self) -> tuple[int, ...]:
        """Shape of chunked image."""
        shape: list[int] = []
        if self.planarconfig == 2 and self.samplesperpixel > 1:
            shape.append(self.samplesperpixel)
        if self.is_tiled:
            if self.imagedepth > 1:
                shape.append(
                    (self.imagedepth + self.tiledepth - 1) // self.tiledepth
                )
            shape.append(
                (self.imagelength + self.tilelength - 1) // self.tilelength
            )
            shape.append(
                (self.imagewidth + self.tilewidth - 1) // self.tilewidth
            )
        else:
            if self.imagedepth > 1:
                shape.append(self.imagedepth)
            shape.append(
                (self.imagelength + self.rowsperstrip - 1) // self.rowsperstrip
            )
            shape.append(1)
        if self.planarconfig == 1 and self.samplesperpixel > 1:
            shape.append(1)
        return tuple(shape)

    @cached_property
    def hash(self) -> int:
        """Checksum to identify pages in same series.

        Pages with the same hash can use the same decode function.
        The hash is calculated from the following properties:
        :py:attr:`TiffFile.tiff`,
        :py:attr:`TiffPage.shaped`,
        :py:attr:`TiffPage.rowsperstrip`,
        :py:attr:`TiffPage.tilewidth`,
        :py:attr:`TiffPage.tilelength`,
        :py:attr:`TiffPage.tiledepth`,
        :py:attr:`TiffPage.sampleformat`,
        :py:attr:`TiffPage.bitspersample`,
        :py:attr:`TiffPage.fillorder`,
        :py:attr:`TiffPage.predictor`,
        :py:attr:`TiffPage.compression`,
        :py:attr:`TiffPage.extrasamples`, and
        :py:attr:`TiffPage.photometric`.

        """
        return hash(
            self.shaped
            + (
                self.parent.tiff,
                self.rowsperstrip,
                self.tilewidth,
                self.tilelength,
                self.tiledepth,
                self.sampleformat,
                self.bitspersample,
                self.fillorder,
                self.predictor,
                self.compression,
                self.extrasamples,
                self.photometric,
            )
        )

    @cached_property
    def pages(self) -> TiffPages | None:
        """Sequence of sub-pages, SubIFDs."""
        if 330 not in self.tags:
            return None
        return TiffPages(self, index=self.index)

    @cached_property
    def maxworkers(self) -> int:
        """Maximum number of threads for decoding segments.

        A value of 0 disables multi-threading also when stacking pages.

        """
        if self.is_contiguous or self.dtype is None:
            return 0
        if self.compression in TIFF.IMAGE_COMPRESSIONS:
            return min(TIFF.MAXWORKERS, len(self.dataoffsets))
        bytecount = product(self.chunks) * self.dtype.itemsize
        if bytecount < 2048:
            # disable multi-threading for small segments
            return 0
        if self.compression == 5 and bytecount < 14336:
            # disable multi-threading for small LZW compressed segments
            return 0
        if len(self.dataoffsets) < 4:
            return 1
        if self.compression != 1 or self.fillorder != 1 or self.predictor != 1:
            if imagecodecs is not None:
                return min(TIFF.MAXWORKERS, len(self.dataoffsets))
        return 2  # optimum for large number of uncompressed tiles

    @cached_property
    def is_contiguous(self) -> bool:
        """Image data is stored contiguously.

        Contiguous image data can be read from
        ``offset=TiffPage.dataoffsets[0]`` with ``size=TiffPage.nbytes``.
        Excludes prediction and fillorder.

        """
        if (
            self.sampleformat == 5
            or self.compression != 1
            or self.bitspersample not in {8, 16, 32, 64}
        ):
            return False
        if 322 in self.tags:  # TileWidth
            if (
                self.imagewidth != self.tilewidth
                or self.imagelength % self.tilelength
                or self.tilewidth % 16
                or self.tilelength % 16
            ):
                return False
            if (
                32997 in self.tags  # ImageDepth
                and 32998 in self.tags  # TileDepth
                and (
                    self.imagelength != self.tilelength
                    or self.imagedepth % self.tiledepth
                )
            ):
                return False
        offsets = self.dataoffsets
        bytecounts = self.databytecounts
        if len(offsets) == 0:
            return False
        if len(offsets) == 1:
            return True
        if self.is_stk or self.is_lsm:
            return True
        if sum(bytecounts) != self.nbytes:
            return False
        if all(
            bytecounts[i] != 0 and offsets[i] + bytecounts[i] == offsets[i + 1]
            for i in range(len(offsets) - 1)
        ):
            return True
        return False

    @cached_property
    def is_final(self) -> bool:
        """Image data are stored in final form. Excludes byte-swapping."""
        return (
            self.is_contiguous
            and self.fillorder == 1
            and self.predictor == 1
            and not self.is_subsampled
        )

    @cached_property
    def is_memmappable(self) -> bool:
        """Image data in file can be memory-mapped to NumPy array."""
        return (
            self.parent.filehandle.is_file
            and self.is_final
            # and (self.bitspersample == 8 or self.parent.isnative)
            # aligned?
            and self.dtype is not None
            and self.dataoffsets[0] % self.dtype.itemsize == 0
        )

    def __repr__(self) -> str:
        index = self._index if len(self._index) > 1 else self._index[0]
        return f'<tifffile.TiffPage {index} @{self.offset}>'

    def __str__(self) -> str:
        return self._str()

    def _str(self, detail: int = 0, width: int = 79) -> str:
        """Return string containing information about TiffPage."""
        if self.keyframe != self:
            return TiffFrame._str(
                self, detail, width  # type: ignore[arg-type]
            )
        attr = ''
        for name in ('memmappable', 'final', 'contiguous'):
            attr = getattr(self, 'is_' + name)
            if attr:
                attr = name.upper()
                break

        def tostr(name: str, /, skip: int = 1) -> str:
            obj = getattr(self, name)
            if obj == skip:
                return ''
            try:
                value = getattr(obj, 'name')
            except AttributeError:
                return ''
            return str(value)

        info = '  '.join(
            s.lower()
            for s in (
                'x'.join(str(i) for i in self.shape),
                f'{SAMPLEFORMAT(self.sampleformat).name}{self.bitspersample}',
                ' '.join(
                    i
                    for i in (
                        PHOTOMETRIC(self.photometric).name,
                        'REDUCED' if self.is_reduced else '',
                        'MASK' if self.is_mask else '',
                        'TILED' if self.is_tiled else '',
                        tostr('compression'),
                        tostr('planarconfig'),
                        tostr('predictor'),
                        tostr('fillorder'),
                    )
                    + (attr,)
                    if i
                ),
                '|'.join(f.upper() for f in sorted(self.flags)),
            )
            if s
        )
        index = self._index if len(self._index) > 1 else self._index[0]
        info = f'TiffPage {index} @{self.offset}  {info}'
        if detail <= 0:
            return info
        info_list = [info, self.tags._str(detail + 1, width=width)]
        if detail > 1:
            for name in ('ndpi',):
                name = name + '_tags'
                attr = getattr(self, name, '')
                if attr:
                    info_list.append(
                        f'{name.upper()}\n'
                        f'{pformat(attr, width=width, height=detail * 8)}'
                    )
        if detail > 3:
            try:
                data = self.asarray()
                info_list.append(
                    f'DATA\n'
                    f'{pformat(data, width=width, height=detail * 8)}'
                )
            except Exception:
                pass
        return '\n\n'.join(info_list)

    def _sample_names(self) -> list[str] | None:
        """Return names of samples."""
        if 'S' not in self.axes:
            return None
        samples = self.shape[self.axes.find('S')]
        extrasamples = len(self.extrasamples)
        if samples < 1 or extrasamples > 2:
            return None
        if self.photometric == 0:
            names = ['WhiteIsZero']
        elif self.photometric == 1:
            names = ['BlackIsZero']
        elif self.photometric == 2:
            names = ['Red', 'Green', 'Blue']
        elif self.photometric == 5:
            names = ['Cyan', 'Magenta', 'Yellow', 'Black']
        elif self.photometric == 6:
            if self.compression in {6, 7, 34892, 33007}:
                # YCBCR -> RGB for JPEG
                names = ['Red', 'Green', 'Blue']
            else:
                names = ['Luma', 'Cb', 'Cr']
        else:
            return None
        if extrasamples > 0:
            names += [enumarg(EXTRASAMPLE, self.extrasamples[0]).name.title()]
        if extrasamples > 1:
            names += [enumarg(EXTRASAMPLE, self.extrasamples[1]).name.title()]
        if len(names) != samples:
            return None
        return names

    @cached_property
    def flags(self) -> set[str]:
        r"""Set of ``is\_\*`` properties that are True."""
        return {
            name.lower()
            for name in TIFF.PAGE_FLAGS
            if getattr(self, 'is_' + name)
        }

    @cached_property
    def andor_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from Andor tags."""
        if not self.is_andor:
            return None
        result = {'Id': self.tags[4864].value}  # AndorId
        for tag in self.tags:  # list(self.tags.values()):
            code = tag.code
            if not 4864 < code < 5031:
                continue
            name = tag.name
            name = name[5:] if len(name) > 5 else name
            result[name] = tag.value
            # del self.tags[code]
        return result

    @cached_property
    def epics_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from EPICS areaDetector tags.

        Use the :py:func:`epics_datetime` function to get a datetime object
        from the epicsTSSec and epicsTSNsec tags.

        """
        if not self.is_epics:
            return None
        result = {}
        for tag in self.tags:  # list(self.tags.values()):
            code = tag.code
            if not 65000 <= code < 65500:
                continue
            value = tag.value
            if code == 65000:
                # not a POSIX timestamp
                # https://github.com/bluesky/area-detector-handlers/issues/20
                result['timeStamp'] = float(value)
            elif code == 65001:
                result['uniqueID'] = int(value)
            elif code == 65002:
                result['epicsTSSec'] = int(value)
            elif code == 65003:
                result['epicsTSNsec'] = int(value)
            else:
                key, value = value.split(':', 1)
                result[key] = astype(value)
            # del self.tags[code]
        return result

    @cached_property
    def ndpi_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from Hamamatsu NDPI tags."""
        # TODO: parse 65449 ini style comments
        if not self.is_ndpi:
            return None
        tags = self.tags
        result = {}
        for name in ('Make', 'Model', 'Software'):
            result[name] = tags[name].value
        for code, name in TIFF.NDPI_TAGS.items():
            if code in tags:
                result[name] = tags[code].value
                # del tags[code]
        if 'McuStarts' in result:
            mcustarts = result['McuStarts']
            if 'McuStartsHighBytes' in result:
                high = result['McuStartsHighBytes'].astype(numpy.uint64)
                high <<= 32
                mcustarts = mcustarts.astype(numpy.uint64)
                mcustarts += high
                del result['McuStartsHighBytes']
            result['McuStarts'] = mcustarts
        return result

    @cached_property
    def geotiff_tags(self) -> dict[str, Any] | None:
        """Consolidated metadata from GeoTIFF tags."""
        if not self.is_geotiff:
            return None
        tags = self.tags

        gkd = tags.valueof(34735)  # GeoKeyDirectoryTag
        if gkd is None or len(gkd) < 2 or gkd[0] != 1:
            logger().warning(f'{self!r} invalid GeoKeyDirectoryTag')
            return {}

        result = {
            'KeyDirectoryVersion': gkd[0],
            'KeyRevision': gkd[1],
            'KeyRevisionMinor': gkd[2],
            # 'NumberOfKeys': gkd[3],
        }
        # deltags = ['GeoKeyDirectoryTag']
        geokeys = TIFF.GEO_KEYS
        geocodes = TIFF.GEO_CODES
        for index in range(gkd[3]):
            try:
                keyid, tagid, count, offset = gkd[
                    4 + index * 4 : index * 4 + 8
                ]
            except Exception as exc:
                logger().warning(
                    f'{self!r} corrupted GeoKeyDirectoryTag '
                    f'raised {exc!r:.128}'
                )
                continue
            if tagid == 0:
                value = offset
            else:
                try:
                    value = tags[tagid].value[offset : offset + count]
                except TiffFileError as exc:
                    logger().warning(
                        f'{self!r} corrupted GeoKeyDirectoryTag {tagid} '
                        f'raised {exc!r:.128}'
                    )
                    continue
                except KeyError as exc:
                    logger().warning(
                        f'{self!r} GeoKeyDirectoryTag {tagid} not found, '
                        f'raised {exc!r:.128}'
                    )
                    continue
                if tagid == 34737 and count > 1 and value[-1] == '|':
                    value = value[:-1]
                value = value if count > 1 else value[0]
            if keyid in geocodes:
                try:
                    value = geocodes[keyid](value)
                except Exception:
                    pass
            try:
                key = geokeys(keyid).name
            except ValueError:
                key = keyid
            result[key] = value

        value = tags.valueof(33920)  # IntergraphMatrixTag
        if value is not None:
            value = numpy.array(value)
            if value.size == 16:
                value = value.reshape((4, 4)).tolist()
            result['IntergraphMatrix'] = value

        value = tags.valueof(33550)  # ModelPixelScaleTag
        if value is not None:
            result['ModelPixelScale'] = numpy.array(value).tolist()

        value = tags.valueof(33922)  # ModelTiepointTag
        if value is not None:
            value = numpy.array(value).reshape((-1, 6)).squeeze().tolist()
            result['ModelTiepoint'] = value

        value = tags.valueof(34264)  # ModelTransformationTag
        if value is not None:
            value = numpy.array(value).reshape((4, 4)).tolist()
            result['ModelTransformation'] = value

        # if 33550 in tags and 33922 in tags:
        #     sx, sy, sz = tags[33550].value  # ModelPixelScaleTag
        #     tiepoints = tags[33922].value  # ModelTiepointTag
        #     transforms = []
        #     for tp in range(0, len(tiepoints), 6):
        #         i, j, k, x, y, z = tiepoints[tp : tp + 6]
        #         transforms.append(
        #             [
        #                 [sx, 0.0, 0.0, x - i * sx],
        #                 [0.0, -sy, 0.0, y + j * sy],
        #                 [0.0, 0.0, sz, z - k * sz],
        #                 [0.0, 0.0, 0.0, 1.0],
        #             ]
        #         )
        #     if len(tiepoints) == 6:
        #         transforms = transforms[0]
        #     result['ModelTransformation'] = transforms

        rpcc = tags.valueof(50844)  # RPCCoefficientTag
        if rpcc is not None:
            result['RPCCoefficient'] = {
                'ERR_BIAS': rpcc[0],
                'ERR_RAND': rpcc[1],
                'LINE_OFF': rpcc[2],
                'SAMP_OFF': rpcc[3],
                'LAT_OFF': rpcc[4],
                'LONG_OFF': rpcc[5],
                'HEIGHT_OFF': rpcc[6],
                'LINE_SCALE': rpcc[7],
                'SAMP_SCALE': rpcc[8],
                'LAT_SCALE': rpcc[9],
                'LONG_SCALE': rpcc[10],
                'HEIGHT_SCALE': rpcc[11],
                'LINE_NUM_COEFF': rpcc[12:33],
                'LINE_DEN_COEFF ': rpcc[33:53],
                'SAMP_NUM_COEFF': rpcc[53:73],
                'SAMP_DEN_COEFF': rpcc[73:],
            }
        return result

    @cached_property
    def shaped_description(self) -> str | None:
        """Description containing array shape if exists, else None."""
        for description in (self.description, self.description1):
            if not description or '"mibi.' in description:
                return None
            if description[:1] == '{' and '"shape":' in description:
                return description
            if description[:6] == 'shape=':
                return description
        return None

    @cached_property
    def imagej_description(self) -> str | None:
        """ImageJ description if exists, else None."""
        for description in (self.description, self.description1):
            if not description:
                return None
            if description[:7] == 'ImageJ=' or description[:7] == 'SCIFIO=':
                return description
        return None

    @cached_property
    def is_jfif(self) -> bool:
        """JPEG compressed segments contain JFIF metadata."""
        if (
            self.compression not in {6, 7, 34892, 33007}
            or len(self.dataoffsets) < 1
            or self.dataoffsets[0] == 0
            or len(self.databytecounts) < 1
            or self.databytecounts[0] < 11
        ):
            return False
        fh = self.parent.filehandle
        fh.seek(self.dataoffsets[0] + 6)
        data = fh.read(4)
        return data == b'JFIF'  # or data == b'Exif'

    @property
    def is_frame(self) -> bool:
        """Object is :py:class:`TiffFrame` instance."""
        return False

    @property
    def is_virtual(self) -> bool:
        """Page does not have IFD structure in file."""
        return False

    @property
    def is_subifd(self) -> bool:
        """Page is SubIFD of another page."""
        return len(self._index) > 1

    @property
    def is_reduced(self) -> bool:
        """Page is reduced image of another image."""
        return bool(self.subfiletype & 0b1)

    @property
    def is_multipage(self) -> bool:
        """Page is part of multi-page image."""
        return bool(self.subfiletype & 0b10)

    @property
    def is_mask(self) -> bool:
        """Page is transparency mask for another image."""
        return bool(self.subfiletype & 0b100)

    @property
    def is_mrc(self) -> bool:
        """Page is part of Mixed Raster Content."""
        return bool(self.subfiletype & 0b1000)

    @property
    def is_tiled(self) -> bool:
        """Page contains tiled image."""
        return self.tilewidth > 0  # return 322 in self.tags  # TileWidth

    @property
    def is_subsampled(self) -> bool:
        """Page contains chroma subsampled image."""
        if self.subsampling is not None:
            return self.subsampling != (1, 1)
        return self.photometric == 6  # YCbCr
        # RGB JPEG usually stored as subsampled YCbCr
        # self.compression == 7
        # and self.photometric == 2
        # and self.planarconfig == 1

    @property
    def is_imagej(self) -> bool:
        """Page contains ImageJ description metadata."""
        return self.imagej_description is not None

    @property
    def is_shaped(self) -> bool:
        """Page contains Tifffile JSON metadata."""
        return self.shaped_description is not None

    @property
    def is_mdgel(self) -> bool:
        """Page contains MDFileTag tag."""
        return (
            37701 not in self.tags  # AgilentBinary
            and 33445 in self.tags  # MDFileTag
        )

    @property
    def is_agilent(self) -> bool:
        """Page contains Agilent Technologies tags."""
        # tag 270 and 285 contain color names
        return 285 in self.tags and 37701 in self.tags  # AgilentBinary

    @property
    def is_mediacy(self) -> bool:
        """Page contains Media Cybernetics Id tag."""
        tag = self.tags.get(50288)  # MC_Id
        try:
            return tag is not None and tag.value[:7] == b'MC TIFF'
        except Exception:
            return False

    @property
    def is_stk(self) -> bool:
        """Page contains UIC1Tag tag."""
        return 33628 in self.tags

    @property
    def is_lsm(self) -> bool:
        """Page contains CZ_LSMINFO tag."""
        return 34412 in self.tags

    @property
    def is_fluoview(self) -> bool:
        """Page contains FluoView MM_STAMP tag."""
        return 34362 in self.tags

    @property
    def is_nih(self) -> bool:
        """Page contains NIHImageHeader tag."""
        return 43314 in self.tags

    @property
    def is_volumetric(self) -> bool:
        """Page contains SGI ImageDepth tag with value > 1."""
        return self.imagedepth > 1

    @property
    def is_vista(self) -> bool:
        """Software tag is 'ISS Vista'."""
        return self.software == 'ISS Vista'

    @property
    def is_metaseries(self) -> bool:
        """Page contains MDS MetaSeries metadata in ImageDescription tag."""
        if self.index != 0 or self.software != 'MetaSeries':
            return False
        d = self.description
        return d.startswith('<MetaData>') and d.endswith('</MetaData>')

    @property
    def is_ome(self) -> bool:
        """Page contains OME-XML in ImageDescription tag."""
        if self.index != 0 or not self.description:
            return False
        return self.description[-10:].strip().endswith('OME>')

    @property
    def is_scn(self) -> bool:
        """Page contains Leica SCN XML in ImageDescription tag."""
        if self.index != 0 or not self.description:
            return False
        return self.description[-10:].strip().endswith('</scn>')

    @property
    def is_micromanager(self) -> bool:
        """Page contains MicroManagerMetadata tag."""
        return 51123 in self.tags

    @property
    def is_andor(self) -> bool:
        """Page contains Andor Technology tags 4864-5030."""
        return 4864 in self.tags

    @property
    def is_pilatus(self) -> bool:
        """Page contains Pilatus tags."""
        return self.software[:8] == 'TVX TIFF' and self.description[:2] == '# '

    @property
    def is_epics(self) -> bool:
        """Page contains EPICS areaDetector tags."""
        return (
            self.description == 'EPICS areaDetector'
            or self.software == 'EPICS areaDetector'
        )

    @property
    def is_tvips(self) -> bool:
        """Page contains TVIPS metadata."""
        return 37706 in self.tags

    @property
    def is_fei(self) -> bool:
        """Page contains FEI_SFEG or FEI_HELIOS tags."""
        return 34680 in self.tags or 34682 in self.tags

    @property
    def is_sem(self) -> bool:
        """Page contains CZ_SEM tag."""
        return 34118 in self.tags

    @property
    def is_svs(self) -> bool:
        """Page contains Aperio metadata."""
        return self.description[:7] == 'Aperio '

    @property
    def is_bif(self) -> bool:
        """Page contains Ventana metadata."""
        try:
            return 700 in self.tags and (
                # avoid reading XMP tag from file at this point
                # b'<iScan' in self.tags[700].value[:4096]
                'Ventana' in self.software
                or self.software[:17] == 'ScanOutputManager'
                or self.description == 'Label Image'
                or self.description == 'Label_Image'
                or self.description == 'Probability_Image'
            )
        except Exception:
            return False

    @property
    def is_scanimage(self) -> bool:
        """Page contains ScanImage metadata."""
        return (
            self.software[:3] == 'SI.'
            or self.description[:6] == 'state.'
            or 'scanimage.SI' in self.description[-256:]
        )

    @property
    def is_indica(self) -> bool:
        """Page contains IndicaLabs metadata."""
        return self.software[:21] == 'IndicaLabsImageWriter'

    @property
    def is_avs(self) -> bool:
        """Page contains Argos AVS XML metadata."""
        try:
            return (
                65000 in self.tags and self.tags.valueof(65000)[:6] == '<Argos'
            )
        except Exception:
            return False

    @property
    def is_qpi(self) -> bool:
        """Page contains PerkinElmer tissue images metadata."""
        # The ImageDescription tag contains XML with a top-level
        # <PerkinElmer-QPI-ImageDescription> element
        return self.software[:15] == 'PerkinElmer-QPI'

    @property
    def is_geotiff(self) -> bool:
        """Page contains GeoTIFF metadata."""
        return 34735 in self.tags  # GeoKeyDirectoryTag

    @property
    def is_gdal(self) -> bool:
        """Page contains GDAL metadata."""
        # startswith '<GDALMetadata>'
        return 42112 in self.tags  # GDAL_METADATA

    @property
    def is_astrotiff(self) -> bool:
        """Page contains AstroTIFF FITS metadata."""
        return (
            self.description[:7] == 'SIMPLE '
            and self.description[-3:] == 'END'
        )

    @property
    def is_streak(self) -> bool:
        """Page contains Hamamatsu streak metadata."""
        return (
            self.description[:1] == '['
            and '],' in self.description[1:32]
            # and self.tags.get(315, '').value[:19] == 'Copyright Hamamatsu'
        )

    @property
    def is_dng(self) -> bool:
        """Page contains DNG metadata."""
        return 50706 in self.tags  # DNGVersion

    @property
    def is_tiffep(self) -> bool:
        """Page contains TIFF/EP metadata."""
        return 37398 in self.tags  # TIFF/EPStandardID

    @property
    def is_sis(self) -> bool:
        """Page contains Olympus SIS metadata."""
        return 33560 in self.tags or 33471 in self.tags

    @property
    def is_ndpi(self) -> bool:
        """Page contains NDPI metadata."""
        return 65420 in self.tags and 271 in self.tags

    @property
    def is_philips(self) -> bool:
        """Page contains Philips DP metadata."""
        return self.software[:10] == 'Philips DP' and self.description[
            -16:
        ].strip().endswith('</DataObject>')

    @property
    def is_eer(self) -> bool:
        """Page contains EER acquisition metadata."""
        return (
            self.parent.is_bigtiff
            and self.compression in {1, 65000, 65001, 65002}
            and 65001 in self.tags
            and self.tags[65001].dtype == 7
        )