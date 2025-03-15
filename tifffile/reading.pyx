
from .format cimport TiffFormat, ByteOrder
from .files cimport FileHandle
from .pages cimport TiffPage
from .series cimport TiffPages

from libc.stdint cimport int64_t

import os
import numpy as np
import struct

from .types import TiffFileError

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

    cdef public TiffPages pages
    """Sequence of pages in TIFF file."""

    cdef FileHandle fh

    def __init__(
        self,
        object file,
        *,
        str mode = None,
        str name = None,
        offset = None,
        size = None,
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
        offset = -1 if offset is None else offset
        size = -1 if size is None else size
        fh = FileHandle(file, mode='rb', name=name, offset=offset, size=size)
        self.fh = fh

        # Read maximum header size
        header = fh.read_at(0, 32)
        if len(header) < 8:
            raise TiffFileError(f'not a TIFF file {header!r}')
        self.tiff_format = TiffFormat.detect_format(header)
        cdef int64_t start_offset = struct.unpack(
                self.tiff_format.offsetformat,
                fh.read_at(self.tiff_format.headersize - self.tiff_format.offsetsize,
                self.tiff_format.offsetsize),
            )[0]
        self.pages = TiffPages.from_file(
            fh,
            self.tiff_format,
            start_offset
        )

    def __del__(self) -> None:
        self.close()

    @property
    def byteorder(self) -> Literal['>', '<']:
        """Byteorder of TIFF file."""
        return '>' if self.tiff_format.byteorder == ByteOrder.II else '<' # TODO: check if this is correct

    @property
    def filehandle(self) -> FileHandle:
        """File handle."""
        return self.fh

    @property
    def filename(self) -> str:
        """Name of file handle."""
        return self.fh.name

    @property
    def fstat(self) -> Any:
        """Status of file handle's descriptor, if any."""
        try:
            return os.fstat(self.fh.fileno())
        except Exception:  # io.UnsupportedOperation
            return None

    def close(self) -> None:
        """Close open file handle(s)."""
        #for tif in self._files.values():
        #    tif.filehandle.close()
        #if self.fh is not None:
        #    self.fh.close()
        #    self.fh = None

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

        Returns:
            Images from specified pages. See `TiffPage.asarray`
            for operations that are applied (or not) to the image data
            stored in the file.

        """
        if series is None:
            series = 0
        if level is None:
            level = 0
        return self.pages[series].asarray(out=out)

    def __enter__(self) -> TiffFile:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
'''
    @property
    def flags(self) -> set[str]:
        """Set of file flags (a potentially expensive operation)."""
        return {
            name.lower()
            for name in TIFF.FILE_FLAGS
            if getattr(self, 'is_' + name)
        }

    @property
    def is_uniform(self) -> bool:
        """File contains uniform series of pages."""
        # the hashes of IFDs 0, 7, and -1 are the same
        pages = self.pages
        try:
            page = self.pages[0]
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

    @property
    def is_ndtiff(self) -> bool:
        """File has NDTiff format."""
        # file should be accompanied by NDTiff.index
        meta = self.micromanager_metadata
        if meta is not None and meta.get('MajorVersion', 0) >= 2:
            self.is_uniform = True
            return True
        return False

    @property
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

    @property
    def is_mdgel(self) -> bool:
        """File has MD Gel format."""
        # side effect: add second page, if exists, to cache
        try:
            ismdgel = (
                self.pages[0].is_mdgel
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
                self.pages[0].is_sis
                and not self.filename.lower().endswith('.vsi')
            )
        except IndexError:
            return False

    @property
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
        # return xml2dict(self.pages[0].description)['OME']
        if self._omexml:
            return self._omexml
        return self.pages[0].description

    @property
    def scn_metadata(self) -> str | None:
        """Leica SCN XML metadata from ImageDescription tag."""
        if not self.is_scn:
            return None
        return self.pages[0].description

    @property
    def philips_metadata(self) -> str | None:
        """Philips DP XML metadata from ImageDescription tag."""
        if not self.is_philips:
            return None
        return self.pages[0].description

    @property
    def indica_metadata(self) -> str | None:
        """IndicaLabs XML metadata from ImageDescription tag."""
        if not self.is_indica:
            return None
        return self.pages[0].description

    @property
    def avs_metadata(self) -> str | None:
        """Argos AVS XML metadata from tag 65000."""
        if not self.is_avs:
            return None
        return self.pages[0].tags.valueof(65000)

    @property
    def lsm_metadata(self) -> dict[str, Any] | None:
        """LSM metadata from CZ_LSMINFO tag."""
        if not self.is_lsm:
            return None
        return self.pages[0].tags.valueof(34412)  # CZ_LSMINFO

    @property
    def stk_metadata(self) -> dict[str, Any] | None:
        """STK metadata from UIC tags."""
        if not self.is_stk:
            return None
        page = self.pages[0]
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

    @property
    def imagej_metadata(self) -> dict[str, Any] | None:
        """ImageJ metadata from ImageDescription and IJMetadata tags."""
        if not self.is_imagej:
            return None
        page = self.pages[0]
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

    @property
    def fluoview_metadata(self) -> dict[str, Any] | None:
        """FluoView metadata from MM_Header and MM_Stamp tags."""
        if not self.is_fluoview:
            return None
        result = {}
        page = self.pages[0]
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
        return self.pages[0].tags.valueof(43314)  # NIHImageHeader

    @property
    def fei_metadata(self) -> dict[str, Any] | None:
        """FEI metadata from SFEG or HELIOS tags."""
        if not self.is_fei:
            return None
        tags = self.pages[0].tags
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
        return self.pages[0].tags.valueof(34118)

    @property
    def sis_metadata(self) -> dict[str, Any] | None:
        """Olympus SIS metadata from OlympusSIS and OlympusINI tags."""
        if not self.pages[0].is_sis:
            return None
        tags = self.pages[0].tags
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

    @property
    def mdgel_metadata(self) -> dict[str, Any] | None:
        """MD-GEL metadata from MDFileTag tags."""
        if not self.is_mdgel:
            return None
        if 33445 in self.pages[0].tags:
            tags = self.pages[0].tags
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
        return self.pages[0].andor_tags

    @property
    def epics_metadata(self) -> dict[str, Any] | None:
        """EPICS metadata from areaDetector tags."""
        return self.pages[0].epics_tags

    @property
    def tvips_metadata(self) -> dict[str, Any] | None:
        """TVIPS metadata from tag."""
        if not self.is_tvips:
            return None
        return self.pages[0].tags.valueof(37706)

    @property
    def metaseries_metadata(self) -> dict[str, Any] | None:
        """MetaSeries metadata from ImageDescription tag of first tag."""
        # TODO: remove this? It is a per page property
        if not self.is_metaseries:
            return None
        return metaseries_description_metadata(self.pages[0].description)

    @property
    def pilatus_metadata(self) -> dict[str, Any] | None:
        """Pilatus metadata from ImageDescription tag."""
        if not self.is_pilatus:
            return None
        return pilatus_description_metadata(self.pages[0].description)

    @property
    def micromanager_metadata(self) -> dict[str, Any] | None:
        """Non-TIFF Micro-Manager metadata."""
        if not self.is_micromanager:
            return None
        return read_micromanager_metadata(self.fh)

    @property
    def gdal_structural_metadata(self) -> dict[str, Any] | None:
        """Non-TIFF GDAL structural metadata."""
        return read_gdal_structural_metadata(self.fh)

    @property
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
            framedata, roidata, version = read_scanimage_metadata(self.fh)
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
        return self.pages[0].geotiff_tags

    @property
    def gdal_metadata(self) -> dict[str, Any] | None:
        """GDAL XML metadata from GDAL_METADATA tag."""
        if not self.is_gdal:
            return None
        return self.pages[0].tags.valueof(42112)

    @property
    def astrotiff_metadata(self) -> dict[str, Any] | None:
        """AstroTIFF metadata from ImageDescription tag."""
        if not self.is_astrotiff:
            return None
        return astrotiff_description_metadata(self.pages[0].description)

    @property
    def streak_metadata(self) -> dict[str, Any] | None:
        """Hamamatsu streak metadata from ImageDescription tag."""
        if not self.is_streak:
            return None
        return streak_description_metadata(
            self.pages[0].description, self.filehandle
        )

    @property
    def eer_metadata(self) -> str | None:
        """EER AcquisitionMetadata XML from tag 65001."""
        if not self.is_eer:
            return None
        value = self.pages[0].tags.valueof(65001)
        return None if value is None else value.decode()

'''