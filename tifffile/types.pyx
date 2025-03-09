#cython: language_level=3
#cython: boundscheck=True
#cython: wraparound=True
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

import enum
from functools import cached_property
import os
#from .tifffile import TiffTagRegistry

class COMPRESSION(enum.IntEnum):
    """Values of Compression tag.

    Compression scheme used on image data.

    """

    NONE = 1
    """No compression (default)."""
    CCITTRLE = 2  # CCITT 1D
    CCITT_T4 = 3  # T4/Group 3 Fax
    CCITT_T6 = 4  # T6/Group 4 Fax
    LZW = 5
    """Lempel-Ziv-Welch."""
    OJPEG = 6  # old-style JPEG
    JPEG = 7
    """New style JPEG."""
    ADOBE_DEFLATE = 8
    """Deflate, aka ZLIB."""
    JBIG_BW = 9  # VC5
    JBIG_COLOR = 10
    JPEG_99 = 99
    KODAK_262 = 262
    JPEGXR_NDPI = 22610
    """JPEG XR (Hammatsu NDPI)."""
    NEXT = 32766
    SONY_ARW = 32767
    PACKED_RAW = 32769
    SAMSUNG_SRW = 32770
    CCIRLEW = 32771  # Word-aligned 1D Huffman compression
    SAMSUNG_SRW2 = 32772
    PACKBITS = 32773
    """PackBits, aka Macintosh RLE."""
    THUNDERSCAN = 32809
    IT8CTPAD = 32895
    IT8LW = 32896
    IT8MP = 32897
    IT8BL = 32898
    PIXARFILM = 32908
    PIXARLOG = 32909
    DEFLATE = 32946
    DCS = 32947
    APERIO_JP2000_YCBC = 33003  # Matrox libraries
    """JPEG 2000 YCbCr (Leica Aperio)."""
    JPEG_2000_LOSSY = 33004
    """Lossy JPEG 2000 (Bio-Formats)."""
    APERIO_JP2000_RGB = 33005  # Kakadu libraries
    """JPEG 2000 RGB (Leica Aperio)."""
    ALT_JPEG = 33007
    """JPEG (Bio-Formats)."""
    # PANASONIC_RAW1 = 34316
    # PANASONIC_RAW2 = 34826
    # PANASONIC_RAW3 = 34828
    # PANASONIC_RAW4 = 34830
    JBIG = 34661
    SGILOG = 34676  # LogLuv32
    SGILOG24 = 34677
    JPEG2000 = 34712
    """JPEG 2000."""
    NIKON_NEF = 34713
    JBIG2 = 34715
    MDI_BINARY = 34718  # Microsoft Document Imaging
    MDI_PROGRESSIVE = 34719  # Microsoft Document Imaging
    MDI_VECTOR = 34720  # Microsoft Document Imaging
    LERC = 34887
    """ESRI Limited Error Raster Compression."""
    JPEG_LOSSY = 34892  # DNG
    LZMA = 34925
    """Lempel-Ziv-Markov chain Algorithm."""
    ZSTD_DEPRECATED = 34926
    WEBP_DEPRECATED = 34927
    PNG = 34933  # Objective Pathology Services
    """Portable Network Graphics (Zoomable Image File format)."""
    JPEGXR = 34934
    """JPEG XR (Zoomable Image File format)."""
    JETRAW = 48124
    """Jetraw by Dotphoton."""
    ZSTD = 50000
    """Zstandard."""
    WEBP = 50001
    """WebP."""
    JPEGXL = 50002  # GDAL
    """JPEG XL."""
    PIXTIFF = 50013
    """ZLIB (Atalasoft)."""
    JPEGXL_DNG = 52546
    """JPEG XL (DNG)."""
    EER_V0 = 65000  # FIXED82 Thermo Fisher Scientific
    EER_V1 = 65001  # FIXED72 Thermo Fisher Scientific
    EER_V2 = 65002  # VARIABLE Thermo Fisher Scientific
    # KODAK_DCR = 65000
    # PENTAX_PEF = 65535

    def __bool__(self) -> bool:
        return self > 1


class PREDICTOR(enum.IntEnum):
    """Values of Predictor tag.

    A mathematical operator that is applied to the image data before
    compression.

    """

    NONE = 1
    """No prediction scheme used (default)."""
    HORIZONTAL = 2
    """Horizontal differencing."""
    FLOATINGPOINT = 3
    """Floating-point horizontal differencing."""
    HORIZONTALX2 = 34892  # DNG
    HORIZONTALX4 = 34893
    FLOATINGPOINTX2 = 34894
    FLOATINGPOINTX4 = 34895

    def __bool__(self) -> bool:
        return self > 1


class PHOTOMETRIC(enum.IntEnum):
    """Values of PhotometricInterpretation tag.

    The color space of the image.

    """

    MINISWHITE = 0
    """For bilevel and grayscale images, 0 is imaged as white."""
    MINISBLACK = 1
    """For bilevel and grayscale images, 0 is imaged as black."""
    RGB = 2
    """Chroma components are Red, Green, Blue."""
    PALETTE = 3
    """Single chroma component is index into colormap."""
    MASK = 4
    SEPARATED = 5
    """Chroma components are Cyan, Magenta, Yellow, and Key (black)."""
    YCBCR = 6
    """Chroma components are Luma, blue-difference, and red-difference."""
    CIELAB = 8
    ICCLAB = 9
    ITULAB = 10
    CFA = 32803
    """Color Filter Array."""
    LOGL = 32844
    LOGLUV = 32845
    LINEAR_RAW = 34892
    DEPTH_MAP = 51177  # DNG 1.5
    SEMANTIC_MASK = 52527  # DNG 1.6


class FILETYPE(enum.IntFlag):
    """Values of NewSubfileType tag.

    A general indication of the kind of the image.

    """

    UNDEFINED = 0
    """Image is full-resolution (default)."""
    REDUCEDIMAGE = 1
    """Image is reduced-resolution version of another image."""
    PAGE = 2
    """Image is single page of multi-page image."""
    MASK = 4
    """Image is transparency mask for another image."""
    MACRO = 8  # Aperio SVS, or DNG Depth map
    """Image is MACRO image (SVS) or depth map for another image (DNG)."""
    ENHANCED = 16  # DNG
    """Image contains enhanced image (DNG)."""
    DNG = 65536  # 65537: Alternative, 65540: Semantic mask


class OFILETYPE(enum.IntEnum):
    """Values of deprecated SubfileType tag."""

    UNDEFINED = 0
    IMAGE = 1  # full-resolution image
    REDUCEDIMAGE = 2  # reduced-resolution image
    PAGE = 3  # single page of multi-page image


class FILLORDER(enum.IntEnum):
    """Values of FillOrder tag.

    The logical order of bits within a byte.

    """

    MSB2LSB = 1
    """Pixel values are stored in higher-order bits of byte (default)."""
    LSB2MSB = 2
    """Pixels values are stored in lower-order bits of byte."""


class ORIENTATION(enum.IntEnum):
    """Values of Orientation tag.

    The orientation of the image with respect to the rows and columns.

    """

    TOPLEFT = 1  # default
    TOPRIGHT = 2
    BOTRIGHT = 3
    BOTLEFT = 4
    LEFTTOP = 5
    RIGHTTOP = 6
    RIGHTBOT = 7
    LEFTBOT = 8


class PLANARCONFIG(enum.IntEnum):
    """Values of PlanarConfiguration tag.

    Specifies how components of each pixel are stored.

    """

    CONTIG = 1
    """Chunky, component values are stored contiguously (default)."""
    SEPARATE = 2
    """Planar, component values are stored in separate planes."""


class RESUNIT(enum.IntEnum):
    """Values of ResolutionUnit tag.

    The unit of measurement for XResolution and YResolution.

    """

    NONE = 1
    """No absolute unit of measurement."""
    INCH = 2
    """Inch (default)."""
    CENTIMETER = 3
    """Centimeter."""
    MILLIMETER = 4
    """Millimeter (DNG)."""
    MICROMETER = 5
    """Micrometer (DNG)."""

    def __bool__(self) -> bool:
        return self > 1


class EXTRASAMPLE(enum.IntEnum):
    """Values of ExtraSamples tag.

    Interpretation of extra components in a pixel.

    """

    UNSPECIFIED = 0
    """Unspecified data."""
    ASSOCALPHA = 1
    """Associated alpha data with premultiplied color."""
    UNASSALPHA = 2
    """Unassociated alpha data."""


class SAMPLEFORMAT(enum.IntEnum):
    """Values of SampleFormat tag.

    Data type of samples in a pixel.

    """

    UINT = 1
    """Unsigned integer."""
    INT = 2
    """Signed integer."""
    IEEEFP = 3
    """IEEE floating-point"""
    VOID = 4
    """Undefined."""
    COMPLEXINT = 5
    """Complex integer."""
    COMPLEXIEEEFP = 6
    """Complex floating-point."""


class CHUNKMODE(enum.IntEnum):
    """ZarrStore chunk modes.

    Specifies how to chunk data in Zarr 2 stores.

    """

    STRILE = 0
    """Chunk is strip or tile."""
    PLANE = 1
    """Chunk is image plane."""
    PAGE = 2
    """Chunk is image in page."""
    FILE = 3
    """Chunk is image in file."""

class _TIFF:
    """Delay-loaded constants, accessible via :py:attr:`TIFF` instance."""

    @cached_property
    def TAGS(self) -> TiffTagRegistry:
        """Registry of TIFF tag codes and names from TIFF6, TIFF/EP, EXIF."""
        # TODO: divide into baseline, exif, private, ... tags
        from .tags import TiffTagRegistry
        return TiffTagRegistry(
            (
                (11, 'ProcessingSoftware'),
                (254, 'NewSubfileType'),
                (255, 'SubfileType'),
                (256, 'ImageWidth'),
                (257, 'ImageLength'),
                (258, 'BitsPerSample'),
                (259, 'Compression'),
                (262, 'PhotometricInterpretation'),
                (263, 'Thresholding'),
                (264, 'CellWidth'),
                (265, 'CellLength'),
                (266, 'FillOrder'),
                (269, 'DocumentName'),
                (270, 'ImageDescription'),
                (271, 'Make'),
                (272, 'Model'),
                (273, 'StripOffsets'),
                (274, 'Orientation'),
                (277, 'SamplesPerPixel'),
                (278, 'RowsPerStrip'),
                (279, 'StripByteCounts'),
                (280, 'MinSampleValue'),
                (281, 'MaxSampleValue'),
                (282, 'XResolution'),
                (283, 'YResolution'),
                (284, 'PlanarConfiguration'),
                (285, 'PageName'),
                (286, 'XPosition'),
                (287, 'YPosition'),
                (288, 'FreeOffsets'),
                (289, 'FreeByteCounts'),
                (290, 'GrayResponseUnit'),
                (291, 'GrayResponseCurve'),
                (292, 'T4Options'),
                (293, 'T6Options'),
                (296, 'ResolutionUnit'),
                (297, 'PageNumber'),
                (300, 'ColorResponseUnit'),
                (301, 'TransferFunction'),
                (305, 'Software'),
                (306, 'DateTime'),
                (315, 'Artist'),
                (316, 'HostComputer'),
                (317, 'Predictor'),
                (318, 'WhitePoint'),
                (319, 'PrimaryChromaticities'),
                (320, 'ColorMap'),
                (321, 'HalftoneHints'),
                (322, 'TileWidth'),
                (323, 'TileLength'),
                (324, 'TileOffsets'),
                (325, 'TileByteCounts'),
                (326, 'BadFaxLines'),
                (327, 'CleanFaxData'),
                (328, 'ConsecutiveBadFaxLines'),
                (330, 'SubIFDs'),
                (332, 'InkSet'),
                (333, 'InkNames'),
                (334, 'NumberOfInks'),
                (336, 'DotRange'),
                (337, 'TargetPrinter'),
                (338, 'ExtraSamples'),
                (339, 'SampleFormat'),
                (340, 'SMinSampleValue'),
                (341, 'SMaxSampleValue'),
                (342, 'TransferRange'),
                (343, 'ClipPath'),
                (344, 'XClipPathUnits'),
                (345, 'YClipPathUnits'),
                (346, 'Indexed'),
                (347, 'JPEGTables'),
                (351, 'OPIProxy'),
                (400, 'GlobalParametersIFD'),
                (401, 'ProfileType'),
                (402, 'FaxProfile'),
                (403, 'CodingMethods'),
                (404, 'VersionYear'),
                (405, 'ModeNumber'),
                (433, 'Decode'),
                (434, 'DefaultImageColor'),
                (435, 'T82Options'),
                (437, 'JPEGTables'),  # 347
                (512, 'JPEGProc'),
                (513, 'JPEGInterchangeFormat'),
                (514, 'JPEGInterchangeFormatLength'),
                (515, 'JPEGRestartInterval'),
                (517, 'JPEGLosslessPredictors'),
                (518, 'JPEGPointTransforms'),
                (519, 'JPEGQTables'),
                (520, 'JPEGDCTables'),
                (521, 'JPEGACTables'),
                (529, 'YCbCrCoefficients'),
                (530, 'YCbCrSubSampling'),
                (531, 'YCbCrPositioning'),
                (532, 'ReferenceBlackWhite'),
                (559, 'StripRowCounts'),
                (700, 'XMP'),  # XMLPacket
                (769, 'GDIGamma'),  # GDI+
                (770, 'ICCProfileDescriptor'),  # GDI+
                (771, 'SRGBRenderingIntent'),  # GDI+
                (800, 'ImageTitle'),  # GDI+
                (907, 'SiffCompress'),  # https://github.com/MaimonLab/SiffPy
                (999, 'USPTO_Miscellaneous'),
                (4864, 'AndorId'),  # TODO, Andor Technology 4864 - 5030
                (4869, 'AndorTemperature'),
                (4876, 'AndorExposureTime'),
                (4878, 'AndorKineticCycleTime'),
                (4879, 'AndorAccumulations'),
                (4881, 'AndorAcquisitionCycleTime'),
                (4882, 'AndorReadoutTime'),
                (4884, 'AndorPhotonCounting'),
                (4885, 'AndorEmDacLevel'),
                (4890, 'AndorFrames'),
                (4896, 'AndorHorizontalFlip'),
                (4897, 'AndorVerticalFlip'),
                (4898, 'AndorClockwise'),
                (4899, 'AndorCounterClockwise'),
                (4904, 'AndorVerticalClockVoltage'),
                (4905, 'AndorVerticalShiftSpeed'),
                (4907, 'AndorPreAmpSetting'),
                (4908, 'AndorCameraSerial'),
                (4911, 'AndorActualTemperature'),
                (4912, 'AndorBaselineClamp'),
                (4913, 'AndorPrescans'),
                (4914, 'AndorModel'),
                (4915, 'AndorChipSizeX'),
                (4916, 'AndorChipSizeY'),
                (4944, 'AndorBaselineOffset'),
                (4966, 'AndorSoftwareVersion'),
                (18246, 'Rating'),
                (18247, 'XP_DIP_XML'),
                (18248, 'StitchInfo'),
                (18249, 'RatingPercent'),
                (20481, 'ResolutionXUnit'),  # GDI+
                (20482, 'ResolutionYUnit'),  # GDI+
                (20483, 'ResolutionXLengthUnit'),  # GDI+
                (20484, 'ResolutionYLengthUnit'),  # GDI+
                (20485, 'PrintFlags'),  # GDI+
                (20486, 'PrintFlagsVersion'),  # GDI+
                (20487, 'PrintFlagsCrop'),  # GDI+
                (20488, 'PrintFlagsBleedWidth'),  # GDI+
                (20489, 'PrintFlagsBleedWidthScale'),  # GDI+
                (20490, 'HalftoneLPI'),  # GDI+
                (20491, 'HalftoneLPIUnit'),  # GDI+
                (20492, 'HalftoneDegree'),  # GDI+
                (20493, 'HalftoneShape'),  # GDI+
                (20494, 'HalftoneMisc'),  # GDI+
                (20495, 'HalftoneScreen'),  # GDI+
                (20496, 'JPEGQuality'),  # GDI+
                (20497, 'GridSize'),  # GDI+
                (20498, 'ThumbnailFormat'),  # GDI+
                (20499, 'ThumbnailWidth'),  # GDI+
                (20500, 'ThumbnailHeight'),  # GDI+
                (20501, 'ThumbnailColorDepth'),  # GDI+
                (20502, 'ThumbnailPlanes'),  # GDI+
                (20503, 'ThumbnailRawBytes'),  # GDI+
                (20504, 'ThumbnailSize'),  # GDI+
                (20505, 'ThumbnailCompressedSize'),  # GDI+
                (20506, 'ColorTransferFunction'),  # GDI+
                (20507, 'ThumbnailData'),
                (20512, 'ThumbnailImageWidth'),  # GDI+
                (20513, 'ThumbnailImageHeight'),  # GDI+
                (20514, 'ThumbnailBitsPerSample'),  # GDI+
                (20515, 'ThumbnailCompression'),
                (20516, 'ThumbnailPhotometricInterp'),  # GDI+
                (20517, 'ThumbnailImageDescription'),  # GDI+
                (20518, 'ThumbnailEquipMake'),  # GDI+
                (20519, 'ThumbnailEquipModel'),  # GDI+
                (20520, 'ThumbnailStripOffsets'),  # GDI+
                (20521, 'ThumbnailOrientation'),  # GDI+
                (20522, 'ThumbnailSamplesPerPixel'),  # GDI+
                (20523, 'ThumbnailRowsPerStrip'),  # GDI+
                (20524, 'ThumbnailStripBytesCount'),  # GDI+
                (20525, 'ThumbnailResolutionX'),
                (20526, 'ThumbnailResolutionY'),
                (20527, 'ThumbnailPlanarConfig'),  # GDI+
                (20528, 'ThumbnailResolutionUnit'),
                (20529, 'ThumbnailTransferFunction'),
                (20530, 'ThumbnailSoftwareUsed'),  # GDI+
                (20531, 'ThumbnailDateTime'),  # GDI+
                (20532, 'ThumbnailArtist'),  # GDI+
                (20533, 'ThumbnailWhitePoint'),  # GDI+
                (20534, 'ThumbnailPrimaryChromaticities'),  # GDI+
                (20535, 'ThumbnailYCbCrCoefficients'),  # GDI+
                (20536, 'ThumbnailYCbCrSubsampling'),  # GDI+
                (20537, 'ThumbnailYCbCrPositioning'),
                (20538, 'ThumbnailRefBlackWhite'),  # GDI+
                (20539, 'ThumbnailCopyRight'),  # GDI+
                (20545, 'InteroperabilityIndex'),
                (20546, 'InteroperabilityVersion'),
                (20624, 'LuminanceTable'),
                (20625, 'ChrominanceTable'),
                (20736, 'FrameDelay'),  # GDI+
                (20737, 'LoopCount'),  # GDI+
                (20738, 'GlobalPalette'),  # GDI+
                (20739, 'IndexBackground'),  # GDI+
                (20740, 'IndexTransparent'),  # GDI+
                (20752, 'PixelUnit'),  # GDI+
                (20753, 'PixelPerUnitX'),  # GDI+
                (20754, 'PixelPerUnitY'),  # GDI+
                (20755, 'PaletteHistogram'),  # GDI+
                (28672, 'SonyRawFileType'),  # Sony ARW
                (28722, 'VignettingCorrParams'),  # Sony ARW
                (28725, 'ChromaticAberrationCorrParams'),  # Sony ARW
                (28727, 'DistortionCorrParams'),  # Sony ARW
                # Private tags >= 32768
                (32781, 'ImageID'),
                (32931, 'WangTag1'),
                (32932, 'WangAnnotation'),
                (32933, 'WangTag3'),
                (32934, 'WangTag4'),
                (32953, 'ImageReferencePoints'),
                (32954, 'RegionXformTackPoint'),
                (32955, 'WarpQuadrilateral'),
                (32956, 'AffineTransformMat'),
                (32995, 'Matteing'),
                (32996, 'DataType'),  # use SampleFormat
                (32997, 'ImageDepth'),
                (32998, 'TileDepth'),
                (33300, 'ImageFullWidth'),
                (33301, 'ImageFullLength'),
                (33302, 'TextureFormat'),
                (33303, 'TextureWrapModes'),
                (33304, 'FieldOfViewCotangent'),
                (33305, 'MatrixWorldToScreen'),
                (33306, 'MatrixWorldToCamera'),
                (33405, 'Model2'),
                (33421, 'CFARepeatPatternDim'),
                (33422, 'CFAPattern'),
                (33423, 'BatteryLevel'),
                (33424, 'KodakIFD'),
                (33434, 'ExposureTime'),
                (33437, 'FNumber'),
                (33432, 'Copyright'),
                (33445, 'MDFileTag'),
                (33446, 'MDScalePixel'),
                (33447, 'MDColorTable'),
                (33448, 'MDLabName'),
                (33449, 'MDSampleInfo'),
                (33450, 'MDPrepDate'),
                (33451, 'MDPrepTime'),
                (33452, 'MDFileUnits'),
                (33465, 'NiffRotation'),  # NIFF
                (33466, 'NiffNavyCompression'),  # NIFF
                (33467, 'NiffTileIndex'),  # NIFF
                (33471, 'OlympusINI'),
                (33550, 'ModelPixelScaleTag'),
                (33560, 'OlympusSIS'),  # see also 33471 and 34853
                (33589, 'AdventScale'),
                (33590, 'AdventRevision'),
                (33628, 'UIC1tag'),  # Metamorph  Universal Imaging Corp STK
                (33629, 'UIC2tag'),
                (33630, 'UIC3tag'),
                (33631, 'UIC4tag'),
                (33723, 'IPTCNAA'),
                (33858, 'ExtendedTagsOffset'),  # DEFF points IFD with tags
                (33918, 'IntergraphPacketData'),  # INGRPacketDataTag
                (33919, 'IntergraphFlagRegisters'),  # INGRFlagRegisters
                (33920, 'IntergraphMatrixTag'),  # IrasBTransformationMatrix
                (33921, 'INGRReserved'),
                (33922, 'ModelTiepointTag'),
                (33923, 'LeicaMagic'),
                (34016, 'Site'),  # 34016..34032 ANSI IT8 TIFF/IT
                (34017, 'ColorSequence'),
                (34018, 'IT8Header'),
                (34019, 'RasterPadding'),
                (34020, 'BitsPerRunLength'),
                (34021, 'BitsPerExtendedRunLength'),
                (34022, 'ColorTable'),
                (34023, 'ImageColorIndicator'),
                (34024, 'BackgroundColorIndicator'),
                (34025, 'ImageColorValue'),
                (34026, 'BackgroundColorValue'),
                (34027, 'PixelIntensityRange'),
                (34028, 'TransparencyIndicator'),
                (34029, 'ColorCharacterization'),
                (34030, 'HCUsage'),
                (34031, 'TrapIndicator'),
                (34032, 'CMYKEquivalent'),
                (34118, 'CZ_SEM'),  # Zeiss SEM
                (34152, 'AFCP_IPTC'),
                (34232, 'PixelMagicJBIGOptions'),  # EXIF, also TI FrameCount
                (34263, 'JPLCartoIFD'),
                (34122, 'IPLAB'),  # number of images
                (34264, 'ModelTransformationTag'),
                (34306, 'WB_GRGBLevels'),  # Leaf MOS
                (34310, 'LeafData'),
                (34361, 'MM_Header'),
                (34362, 'MM_Stamp'),
                (34363, 'MM_Unknown'),
                (34377, 'ImageResources'),  # Photoshop
                (34386, 'MM_UserBlock'),
                (34412, 'CZ_LSMINFO'),
                (34665, 'ExifTag'),
                (34675, 'InterColorProfile'),  # ICCProfile
                (34680, 'FEI_SFEG'),  #
                (34682, 'FEI_HELIOS'),  #
                (34683, 'FEI_TITAN'),  #
                (34687, 'FXExtensions'),
                (34688, 'MultiProfiles'),
                (34689, 'SharedData'),
                (34690, 'T88Options'),
                (34710, 'MarCCD'),  # offset to MarCCD header
                (34732, 'ImageLayer'),
                (34735, 'GeoKeyDirectoryTag'),
                (34736, 'GeoDoubleParamsTag'),
                (34737, 'GeoAsciiParamsTag'),
                (34750, 'JBIGOptions'),
                (34821, 'PIXTIFF'),  # ? Pixel Translations Inc
                (34850, 'ExposureProgram'),
                (34852, 'SpectralSensitivity'),
                (34853, 'GPSTag'),  # GPSIFD  also OlympusSIS2
                (34853, 'OlympusSIS2'),
                (34855, 'ISOSpeedRatings'),
                (34855, 'PhotographicSensitivity'),
                (34856, 'OECF'),  # optoelectric conversion factor
                (34857, 'Interlace'),  # TIFF/EP
                (34858, 'TimeZoneOffset'),  # TIFF/EP
                (34859, 'SelfTimerMode'),  # TIFF/EP
                (34864, 'SensitivityType'),
                (34865, 'StandardOutputSensitivity'),
                (34866, 'RecommendedExposureIndex'),
                (34867, 'ISOSpeed'),
                (34868, 'ISOSpeedLatitudeyyy'),
                (34869, 'ISOSpeedLatitudezzz'),
                (34908, 'HylaFAXFaxRecvParams'),
                (34909, 'HylaFAXFaxSubAddress'),
                (34910, 'HylaFAXFaxRecvTime'),
                (34911, 'FaxDcs'),
                (34929, 'FedexEDR'),
                (34954, 'LeafSubIFD'),
                (34959, 'Aphelion1'),
                (34960, 'Aphelion2'),
                (34961, 'AphelionInternal'),  # ADCIS
                (36864, 'ExifVersion'),
                (36867, 'DateTimeOriginal'),
                (36868, 'DateTimeDigitized'),
                (36873, 'GooglePlusUploadCode'),
                (36880, 'OffsetTime'),
                (36881, 'OffsetTimeOriginal'),
                (36882, 'OffsetTimeDigitized'),
                # TODO, Pilatus/CHESS/TV6 36864..37120 conflicting with Exif
                (36864, 'TVX_Unknown'),
                (36865, 'TVX_NumExposure'),
                (36866, 'TVX_NumBackground'),
                (36867, 'TVX_ExposureTime'),
                (36868, 'TVX_BackgroundTime'),
                (36870, 'TVX_Unknown'),
                (36873, 'TVX_SubBpp'),
                (36874, 'TVX_SubWide'),
                (36875, 'TVX_SubHigh'),
                (36876, 'TVX_BlackLevel'),
                (36877, 'TVX_DarkCurrent'),
                (36878, 'TVX_ReadNoise'),
                (36879, 'TVX_DarkCurrentNoise'),
                (36880, 'TVX_BeamMonitor'),
                (37120, 'TVX_UserVariables'),  # A/D values
                (37121, 'ComponentsConfiguration'),
                (37122, 'CompressedBitsPerPixel'),
                (37377, 'ShutterSpeedValue'),
                (37378, 'ApertureValue'),
                (37379, 'BrightnessValue'),
                (37380, 'ExposureBiasValue'),
                (37381, 'MaxApertureValue'),
                (37382, 'SubjectDistance'),
                (37383, 'MeteringMode'),
                (37384, 'LightSource'),
                (37385, 'Flash'),
                (37386, 'FocalLength'),
                (37387, 'FlashEnergy'),  # TIFF/EP
                (37388, 'SpatialFrequencyResponse'),  # TIFF/EP
                (37389, 'Noise'),  # TIFF/EP
                (37390, 'FocalPlaneXResolution'),  # TIFF/EP
                (37391, 'FocalPlaneYResolution'),  # TIFF/EP
                (37392, 'FocalPlaneResolutionUnit'),  # TIFF/EP
                (37393, 'ImageNumber'),  # TIFF/EP
                (37394, 'SecurityClassification'),  # TIFF/EP
                (37395, 'ImageHistory'),  # TIFF/EP
                (37396, 'SubjectLocation'),  # TIFF/EP
                (37397, 'ExposureIndex'),  # TIFF/EP
                (37398, 'TIFFEPStandardID'),  # TIFF/EP
                (37399, 'SensingMethod'),  # TIFF/EP
                (37434, 'CIP3DataFile'),
                (37435, 'CIP3Sheet'),
                (37436, 'CIP3Side'),
                (37439, 'StoNits'),
                (37500, 'MakerNote'),
                (37510, 'UserComment'),
                (37520, 'SubsecTime'),
                (37521, 'SubsecTimeOriginal'),
                (37522, 'SubsecTimeDigitized'),
                (37679, 'MODIText'),  # Microsoft Office Document Imaging
                (37680, 'MODIOLEPropertySetStorage'),
                (37681, 'MODIPositioning'),
                (37701, 'AgilentBinary'),  # private structure
                (37702, 'AgilentString'),  # file description
                (37706, 'TVIPS'),  # offset to TemData structure
                (37707, 'TVIPS1'),
                (37708, 'TVIPS2'),  # same TemData structure as undefined
                (37724, 'ImageSourceData'),  # Photoshop
                (37888, 'Temperature'),
                (37889, 'Humidity'),
                (37890, 'Pressure'),
                (37891, 'WaterDepth'),
                (37892, 'Acceleration'),
                (37893, 'CameraElevationAngle'),
                (40000, 'XPos'),  # Janelia
                (40001, 'YPos'),
                (40002, 'ZPos'),
                (40001, 'MC_IpWinScal'),  # Media Cybernetics
                (40001, 'RecipName'),  # MS FAX
                (40002, 'RecipNumber'),
                (40003, 'SenderName'),
                (40004, 'Routing'),
                (40005, 'CallerId'),
                (40006, 'TSID'),
                (40007, 'CSID'),
                (40008, 'FaxTime'),
                (40100, 'MC_IdOld'),
                (40106, 'MC_Unknown'),
                (40965, 'InteroperabilityTag'),  # InteropOffset
                (40091, 'XPTitle'),
                (40092, 'XPComment'),
                (40093, 'XPAuthor'),
                (40094, 'XPKeywords'),
                (40095, 'XPSubject'),
                (40960, 'FlashpixVersion'),
                (40961, 'ColorSpace'),
                (40962, 'PixelXDimension'),
                (40963, 'PixelYDimension'),
                (40964, 'RelatedSoundFile'),
                (40976, 'SamsungRawPointersOffset'),
                (40977, 'SamsungRawPointersLength'),
                (41217, 'SamsungRawByteOrder'),
                (41218, 'SamsungRawUnknown'),
                (41483, 'FlashEnergy'),
                (41484, 'SpatialFrequencyResponse'),
                (41485, 'Noise'),  # 37389
                (41486, 'FocalPlaneXResolution'),  # 37390
                (41487, 'FocalPlaneYResolution'),  # 37391
                (41488, 'FocalPlaneResolutionUnit'),  # 37392
                (41489, 'ImageNumber'),  # 37393
                (41490, 'SecurityClassification'),  # 37394
                (41491, 'ImageHistory'),  # 37395
                (41492, 'SubjectLocation'),  # 37395
                (41493, 'ExposureIndex '),  # 37397
                (41494, 'TIFF-EPStandardID'),
                (41495, 'SensingMethod'),  # 37399
                (41728, 'FileSource'),
                (41729, 'SceneType'),
                (41730, 'CFAPattern'),  # 33422
                (41985, 'CustomRendered'),
                (41986, 'ExposureMode'),
                (41987, 'WhiteBalance'),
                (41988, 'DigitalZoomRatio'),
                (41989, 'FocalLengthIn35mmFilm'),
                (41990, 'SceneCaptureType'),
                (41991, 'GainControl'),
                (41992, 'Contrast'),
                (41993, 'Saturation'),
                (41994, 'Sharpness'),
                (41995, 'DeviceSettingDescription'),
                (41996, 'SubjectDistanceRange'),
                (42016, 'ImageUniqueID'),
                (42032, 'CameraOwnerName'),
                (42033, 'BodySerialNumber'),
                (42034, 'LensSpecification'),
                (42035, 'LensMake'),
                (42036, 'LensModel'),
                (42037, 'LensSerialNumber'),
                (42080, 'CompositeImage'),
                (42081, 'SourceImageNumberCompositeImage'),
                (42082, 'SourceExposureTimesCompositeImage'),
                (42112, 'GDAL_METADATA'),
                (42113, 'GDAL_NODATA'),
                (42240, 'Gamma'),
                (43314, 'NIHImageHeader'),
                (44992, 'ExpandSoftware'),
                (44993, 'ExpandLens'),
                (44994, 'ExpandFilm'),
                (44995, 'ExpandFilterLens'),
                (44996, 'ExpandScanner'),
                (44997, 'ExpandFlashLamp'),
                (48129, 'PixelFormat'),  # HDP and WDP
                (48130, 'Transformation'),
                (48131, 'Uncompressed'),
                (48132, 'ImageType'),
                (48256, 'ImageWidth'),  # 256
                (48257, 'ImageHeight'),
                (48258, 'WidthResolution'),
                (48259, 'HeightResolution'),
                (48320, 'ImageOffset'),
                (48321, 'ImageByteCount'),
                (48322, 'AlphaOffset'),
                (48323, 'AlphaByteCount'),
                (48324, 'ImageDataDiscard'),
                (48325, 'AlphaDataDiscard'),
                (50003, 'KodakAPP3'),
                (50215, 'OceScanjobDescription'),
                (50216, 'OceApplicationSelector'),
                (50217, 'OceIdentificationNumber'),
                (50218, 'OceImageLogicCharacteristics'),
                (50255, 'Annotations'),
                (50288, 'MC_Id'),  # Media Cybernetics
                (50289, 'MC_XYPosition'),
                (50290, 'MC_ZPosition'),
                (50291, 'MC_XYCalibration'),
                (50292, 'MC_LensCharacteristics'),
                (50293, 'MC_ChannelName'),
                (50294, 'MC_ExcitationWavelength'),
                (50295, 'MC_TimeStamp'),
                (50296, 'MC_FrameProperties'),
                (50341, 'PrintImageMatching'),
                (50495, 'PCO_RAW'),  # TODO, PCO CamWare
                (50547, 'OriginalFileName'),
                (50560, 'USPTO_OriginalContentType'),  # US Patent Office
                (50561, 'USPTO_RotationCode'),
                (50648, 'CR2Unknown1'),
                (50649, 'CR2Unknown2'),
                (50656, 'CR2CFAPattern'),
                (50674, 'LercParameters'),  # ESGI 50674 .. 50677
                (50706, 'DNGVersion'),  # DNG 50706 .. 51114
                (50707, 'DNGBackwardVersion'),
                (50708, 'UniqueCameraModel'),
                (50709, 'LocalizedCameraModel'),
                (50710, 'CFAPlaneColor'),
                (50711, 'CFALayout'),
                (50712, 'LinearizationTable'),
                (50713, 'BlackLevelRepeatDim'),
                (50714, 'BlackLevel'),
                (50715, 'BlackLevelDeltaH'),
                (50716, 'BlackLevelDeltaV'),
                (50717, 'WhiteLevel'),
                (50718, 'DefaultScale'),
                (50719, 'DefaultCropOrigin'),
                (50720, 'DefaultCropSize'),
                (50721, 'ColorMatrix1'),
                (50722, 'ColorMatrix2'),
                (50723, 'CameraCalibration1'),
                (50724, 'CameraCalibration2'),
                (50725, 'ReductionMatrix1'),
                (50726, 'ReductionMatrix2'),
                (50727, 'AnalogBalance'),
                (50728, 'AsShotNeutral'),
                (50729, 'AsShotWhiteXY'),
                (50730, 'BaselineExposure'),
                (50731, 'BaselineNoise'),
                (50732, 'BaselineSharpness'),
                (50733, 'BayerGreenSplit'),
                (50734, 'LinearResponseLimit'),
                (50735, 'CameraSerialNumber'),
                (50736, 'LensInfo'),
                (50737, 'ChromaBlurRadius'),
                (50738, 'AntiAliasStrength'),
                (50739, 'ShadowScale'),
                (50740, 'DNGPrivateData'),
                (50741, 'MakerNoteSafety'),
                (50752, 'RawImageSegmentation'),
                (50778, 'CalibrationIlluminant1'),
                (50779, 'CalibrationIlluminant2'),
                (50780, 'BestQualityScale'),
                (50781, 'RawDataUniqueID'),
                (50784, 'AliasLayerMetadata'),
                (50827, 'OriginalRawFileName'),
                (50828, 'OriginalRawFileData'),
                (50829, 'ActiveArea'),
                (50830, 'MaskedAreas'),
                (50831, 'AsShotICCProfile'),
                (50832, 'AsShotPreProfileMatrix'),
                (50833, 'CurrentICCProfile'),
                (50834, 'CurrentPreProfileMatrix'),
                (50838, 'IJMetadataByteCounts'),
                (50839, 'IJMetadata'),
                (50844, 'RPCCoefficientTag'),
                (50879, 'ColorimetricReference'),
                (50885, 'SRawType'),
                (50898, 'PanasonicTitle'),
                (50899, 'PanasonicTitle2'),
                (50908, 'RSID'),  # DGIWG
                (50909, 'GEO_METADATA'),  # DGIWG XML
                (50931, 'CameraCalibrationSignature'),
                (50932, 'ProfileCalibrationSignature'),
                (50933, 'ProfileIFD'),  # EXTRACAMERAPROFILES
                (50934, 'AsShotProfileName'),
                (50935, 'NoiseReductionApplied'),
                (50936, 'ProfileName'),
                (50937, 'ProfileHueSatMapDims'),
                (50938, 'ProfileHueSatMapData1'),
                (50939, 'ProfileHueSatMapData2'),
                (50940, 'ProfileToneCurve'),
                (50941, 'ProfileEmbedPolicy'),
                (50942, 'ProfileCopyright'),
                (50964, 'ForwardMatrix1'),
                (50965, 'ForwardMatrix2'),
                (50966, 'PreviewApplicationName'),
                (50967, 'PreviewApplicationVersion'),
                (50968, 'PreviewSettingsName'),
                (50969, 'PreviewSettingsDigest'),
                (50970, 'PreviewColorSpace'),
                (50971, 'PreviewDateTime'),
                (50972, 'RawImageDigest'),
                (50973, 'OriginalRawFileDigest'),
                (50974, 'SubTileBlockSize'),
                (50975, 'RowInterleaveFactor'),
                (50981, 'ProfileLookTableDims'),
                (50982, 'ProfileLookTableData'),
                (51008, 'OpcodeList1'),
                (51009, 'OpcodeList2'),
                (51022, 'OpcodeList3'),
                (51023, 'FibicsXML'),  #
                (51041, 'NoiseProfile'),
                (51043, 'TimeCodes'),
                (51044, 'FrameRate'),
                (51058, 'TStop'),
                (51081, 'ReelName'),
                (51089, 'OriginalDefaultFinalSize'),
                (51090, 'OriginalBestQualitySize'),
                (51091, 'OriginalDefaultCropSize'),
                (51105, 'CameraLabel'),
                (51107, 'ProfileHueSatMapEncoding'),
                (51108, 'ProfileLookTableEncoding'),
                (51109, 'BaselineExposureOffset'),
                (51110, 'DefaultBlackRender'),
                (51111, 'NewRawImageDigest'),
                (51112, 'RawToPreviewGain'),
                (51113, 'CacheBlob'),
                (51114, 'CacheVersion'),
                (51123, 'MicroManagerMetadata'),
                (51125, 'DefaultUserCrop'),
                (51159, 'ZIFmetadata'),  # Objective Pathology Services
                (51160, 'ZIFannotations'),  # Objective Pathology Services
                (51177, 'DepthFormat'),
                (51178, 'DepthNear'),
                (51179, 'DepthFar'),
                (51180, 'DepthUnits'),
                (51181, 'DepthMeasureType'),
                (51182, 'EnhanceParams'),
                (52525, 'ProfileGainTableMap'),  # DNG 1.6
                (52526, 'SemanticName'),  # DNG 1.6
                (52528, 'SemanticInstanceID'),  # DNG 1.6
                (52536, 'MaskSubArea'),  # DNG 1.6
                (52543, 'RGBTables'),  # DNG 1.6
                (52529, 'CalibrationIlluminant3'),  # DNG 1.6
                (52531, 'ColorMatrix3'),  # DNG 1.6
                (52530, 'CameraCalibration3'),  # DNG 1.6
                (52538, 'ReductionMatrix3'),  # DNG 1.6
                (52537, 'ProfileHueSatMapData3'),  # DNG 1.6
                (52532, 'ForwardMatrix3'),  # DNG 1.6
                (52533, 'IlluminantData1'),  # DNG 1.6
                (52534, 'IlluminantData2'),  # DNG 1.6
                (53535, 'IlluminantData3'),  # DNG 1.6
                (52544, 'ProfileGainTableMap2'),  # DNG 1.7
                (52547, 'ColumnInterleaveFactor'),  # DNG 1.7
                (52548, 'ImageSequenceInfo'),  # DNG 1.7
                (52550, 'ImageStats'),  # DNG 1.7
                (52551, 'ProfileDynamicRange'),  # DNG 1.7
                (52552, 'ProfileGroupName'),  # DNG 1.7
                (52553, 'JXLDistance'),  # DNG 1.7
                (52554, 'JXLEffort'),  # DNG 1.7
                (52555, 'JXLDecodeSpeed'),  # DNG 1.7
                (55000, 'AperioUnknown55000'),
                (55001, 'AperioMagnification'),
                (55002, 'AperioMPP'),
                (55003, 'AperioScanScopeID'),
                (55004, 'AperioDate'),
                (59932, 'Padding'),
                (59933, 'OffsetSchema'),
                # Reusable Tags 65000-65535
                # (65000, 'DimapDocumentXML'),
                # EER metadata:
                # (65001, 'AcquisitionMetadata'),
                # (65002, 'FrameMetadata'),
                # (65006, 'ImageMetadata'),
                # (65007, 'PosSkipBits'),
                # (65008, 'HorzSubBits'),
                # (65009, 'VertSubBits'),
                # Photoshop Camera RAW EXIF tags:
                # (65000, 'OwnerName'),
                # (65001, 'SerialNumber'),
                # (65002, 'Lens'),
                # (65024, 'KodakKDCPrivateIFD'),
                # (65100, 'RawFile'),
                # (65101, 'Converter'),
                # (65102, 'WhiteBalance'),
                # (65105, 'Exposure'),
                # (65106, 'Shadows'),
                # (65107, 'Brightness'),
                # (65108, 'Contrast'),
                # (65109, 'Saturation'),
                # (65110, 'Sharpness'),
                # (65111, 'Smoothness'),
                # (65112, 'MoireFilter'),
                (65200, 'FlexXML'),
            )
        )

    @cached_property
    def TAG_ATTRIBUTES(self) -> dict[int, str]:
        # map tag codes to TiffPage attribute names
        return {
            254: 'subfiletype',
            256: 'imagewidth',
            257: 'imagelength',
            # 258: 'bitspersample',  # set manually
            259: 'compression',
            262: 'photometric',
            266: 'fillorder',
            270: 'description',
            277: 'samplesperpixel',
            278: 'rowsperstrip',
            284: 'planarconfig',
            # 301: 'transferfunction',  # delay load
            305: 'software',
            # 320: 'colormap',  # delay load
            317: 'predictor',
            322: 'tilewidth',
            323: 'tilelength',
            330: 'subifds',
            338: 'extrasamples',
            # 339: 'sampleformat',  # set manually
            347: 'jpegtables',
            530: 'subsampling',
            32997: 'imagedepth',
            32998: 'tiledepth',
        }


    @cached_property
    def TAG_ENUM(self) -> dict[int, type[enum.Enum]]:
        # map tag codes to Enums
        return {
            254: FILETYPE,
            255: OFILETYPE,
            259: COMPRESSION,
            262: PHOTOMETRIC,
            # 263: THRESHOLD,
            266: FILLORDER,
            274: ORIENTATION,
            284: PLANARCONFIG,
            # 290: GRAYRESPONSEUNIT,
            # 292: GROUP3OPT
            # 293: GROUP4OPT
            296: RESUNIT,
            # 300: COLORRESPONSEUNIT,
            317: PREDICTOR,
            338: EXTRASAMPLE,
            339: SAMPLEFORMAT,
            # 512: JPEGPROC
            # 531: YCBCRPOSITION
        }


    @cached_property
    def EXIF_TAGS(self) -> TiffTagRegistry:
        """Registry of EXIF tags, including private Photoshop Camera RAW."""
        # 65000 - 65112  Photoshop Camera RAW EXIF tags
        from .tags import TiffTagRegistry
        tags = TiffTagRegistry(
            (
                (65000, 'OwnerName'),
                (65001, 'SerialNumber'),
                (65002, 'Lens'),
                (65100, 'RawFile'),
                (65101, 'Converter'),
                (65102, 'WhiteBalance'),
                (65105, 'Exposure'),
                (65106, 'Shadows'),
                (65107, 'Brightness'),
                (65108, 'Contrast'),
                (65109, 'Saturation'),
                (65110, 'Sharpness'),
                (65111, 'Smoothness'),
                (65112, 'MoireFilter'),
            )
        )
        tags.update(TIFF.TAGS)
        return tags

    @cached_property
    def NDPI_TAGS(self) -> TiffTagRegistry:
        """Registry of private TIFF tags for Hamamatsu NDPI (65420-65458)."""
        # TODO: obtain specification
        from .tags import TiffTagRegistry
        return TiffTagRegistry(
            (
                (65324, 'OffsetHighBytes'),
                (65325, 'ByteCountHighBytes'),
                (65420, 'FileFormat'),
                (65421, 'Magnification'),  # SourceLens
                (65422, 'XOffsetFromSlideCenter'),
                (65423, 'YOffsetFromSlideCenter'),
                (65424, 'ZOffsetFromSlideCenter'),  # FocalPlane
                (65425, 'TissueIndex'),
                (65426, 'McuStarts'),
                (65427, 'SlideLabel'),
                (65428, 'AuthCode'),  # ?
                (65429, '65429'),
                (65430, '65430'),
                (65431, '65431'),
                (65432, 'McuStartsHighBytes'),
                (65433, '65433'),
                (65434, 'Fluorescence'),  # FilterSetName, Channel
                (65435, 'ExposureRatio'),
                (65436, 'RedMultiplier'),
                (65437, 'GreenMultiplier'),
                (65438, 'BlueMultiplier'),
                (65439, 'FocusPoints'),
                (65440, 'FocusPointRegions'),
                (65441, 'CaptureMode'),
                (65442, 'ScannerSerialNumber'),
                (65443, '65443'),
                (65444, 'JpegQuality'),
                (65445, 'RefocusInterval'),
                (65446, 'FocusOffset'),
                (65447, 'BlankLines'),
                (65448, 'FirmwareVersion'),
                (65449, 'Comments'),  # PropertyMap, CalibrationInfo
                (65450, 'LabelObscured'),
                (65451, 'Wavelength'),
                (65452, '65452'),
                (65453, 'LampAge'),
                (65454, 'ExposureTime'),
                (65455, 'FocusTime'),
                (65456, 'ScanTime'),
                (65457, 'WriteTime'),
                (65458, 'FullyAutoFocus'),
                (65500, 'DefaultGamma'),
            )
        )

    @cached_property
    def GPS_TAGS(self) -> TiffTagRegistry:
        """Registry of GPS IFD tags."""
        from .tags import TiffTagRegistry
        return TiffTagRegistry(
            (
                (0, 'GPSVersionID'),
                (1, 'GPSLatitudeRef'),
                (2, 'GPSLatitude'),
                (3, 'GPSLongitudeRef'),
                (4, 'GPSLongitude'),
                (5, 'GPSAltitudeRef'),
                (6, 'GPSAltitude'),
                (7, 'GPSTimeStamp'),
                (8, 'GPSSatellites'),
                (9, 'GPSStatus'),
                (10, 'GPSMeasureMode'),
                (11, 'GPSDOP'),
                (12, 'GPSSpeedRef'),
                (13, 'GPSSpeed'),
                (14, 'GPSTrackRef'),
                (15, 'GPSTrack'),
                (16, 'GPSImgDirectionRef'),
                (17, 'GPSImgDirection'),
                (18, 'GPSMapDatum'),
                (19, 'GPSDestLatitudeRef'),
                (20, 'GPSDestLatitude'),
                (21, 'GPSDestLongitudeRef'),
                (22, 'GPSDestLongitude'),
                (23, 'GPSDestBearingRef'),
                (24, 'GPSDestBearing'),
                (25, 'GPSDestDistanceRef'),
                (26, 'GPSDestDistance'),
                (27, 'GPSProcessingMethod'),
                (28, 'GPSAreaInformation'),
                (29, 'GPSDateStamp'),
                (30, 'GPSDifferential'),
                (31, 'GPSHPositioningError'),
            )
        )

    @cached_property
    def IOP_TAGS(self) -> TiffTagRegistry:
        """Registry of Interoperability IFD tags."""
        from .tags import TiffTagRegistry
        return TiffTagRegistry(
            (
                (1, 'InteroperabilityIndex'),
                (2, 'InteroperabilityVersion'),
                (4096, 'RelatedImageFileFormat'),
                (4097, 'RelatedImageWidth'),
                (4098, 'RelatedImageLength'),
            )
        )

    @cached_property
    def PHOTOMETRIC_SAMPLES(self) -> dict[int, int]:
        """Map :py:class:`PHOTOMETRIC` to number of photometric samples."""
        return {
            0: 1,  # MINISWHITE
            1: 1,  # MINISBLACK
            2: 3,  # RGB
            3: 1,  # PALETTE
            4: 1,  # MASK
            5: 4,  # SEPARATED
            6: 3,  # YCBCR
            8: 3,  # CIELAB
            9: 3,  # ICCLAB
            10: 3,  # ITULAB
            32803: 1,  # CFA
            32844: 1,  # LOGL ?
            32845: 3,  # LOGLUV
            34892: 3,  # LINEAR_RAW ?
            51177: 1,  # DEPTH_MAP ?
            52527: 1,  # SEMANTIC_MASK ?
        }

    @cached_property
    def DATA_FORMATS(self) -> dict[int, str]:
        """Map :py:class:`DATATYPE` to Python struct formats."""
        return {
            1: '1B',
            2: '1s',
            3: '1H',
            4: '1I',
            5: '2I',
            6: '1b',
            7: '1B',
            8: '1h',
            9: '1i',
            10: '2i',
            11: '1f',
            12: '1d',
            13: '1I',
            # 14: '',
            # 15: '',
            16: '1Q',
            17: '1q',
            18: '1Q',
        }

    @cached_property
    def DATA_DTYPES(self) -> dict[str, int]:
        """Map NumPy dtype to :py:class:`DATATYPE`."""
        return {
            'B': 1,
            's': 2,
            'H': 3,
            'I': 4,
            '2I': 5,
            'b': 6,
            'h': 8,
            'i': 9,
            '2i': 10,
            'f': 11,
            'd': 12,
            'Q': 16,
            'q': 17,
        }

    @cached_property
    def SAMPLE_DTYPES(self) -> dict[tuple[int, int | tuple[int, ...]], str]:
        """Map :py:class:`SAMPLEFORMAT` and BitsPerSample to NumPy dtype."""
        return {
            # UINT
            (1, 1): '?',  # bitmap
            (1, 2): 'B',
            (1, 3): 'B',
            (1, 4): 'B',
            (1, 5): 'B',
            (1, 6): 'B',
            (1, 7): 'B',
            (1, 8): 'B',
            (1, 9): 'H',
            (1, 10): 'H',
            (1, 11): 'H',
            (1, 12): 'H',
            (1, 13): 'H',
            (1, 14): 'H',
            (1, 15): 'H',
            (1, 16): 'H',
            (1, 17): 'I',
            (1, 18): 'I',
            (1, 19): 'I',
            (1, 20): 'I',
            (1, 21): 'I',
            (1, 22): 'I',
            (1, 23): 'I',
            (1, 24): 'I',
            (1, 25): 'I',
            (1, 26): 'I',
            (1, 27): 'I',
            (1, 28): 'I',
            (1, 29): 'I',
            (1, 30): 'I',
            (1, 31): 'I',
            (1, 32): 'I',
            (1, 64): 'Q',
            # VOID : treat as UINT
            (4, 1): '?',  # bitmap
            (4, 2): 'B',
            (4, 3): 'B',
            (4, 4): 'B',
            (4, 5): 'B',
            (4, 6): 'B',
            (4, 7): 'B',
            (4, 8): 'B',
            (4, 9): 'H',
            (4, 10): 'H',
            (4, 11): 'H',
            (4, 12): 'H',
            (4, 13): 'H',
            (4, 14): 'H',
            (4, 15): 'H',
            (4, 16): 'H',
            (4, 17): 'I',
            (4, 18): 'I',
            (4, 19): 'I',
            (4, 20): 'I',
            (4, 21): 'I',
            (4, 22): 'I',
            (4, 23): 'I',
            (4, 24): 'I',
            (4, 25): 'I',
            (4, 26): 'I',
            (4, 27): 'I',
            (4, 28): 'I',
            (4, 29): 'I',
            (4, 30): 'I',
            (4, 31): 'I',
            (4, 32): 'I',
            (4, 64): 'Q',
            # INT
            (2, 8): 'b',
            (2, 16): 'h',
            (2, 32): 'i',
            (2, 64): 'q',
            # IEEEFP
            (3, 16): 'e',
            (3, 24): 'f',  # float24 bit not supported by numpy
            (3, 32): 'f',
            (3, 64): 'd',
            # COMPLEXIEEEFP
            (6, 64): 'F',
            (6, 128): 'D',
            # RGB565
            (1, (5, 6, 5)): 'B',
            # COMPLEXINT : not supported by numpy
            (5, 16): 'E',
            (5, 32): 'F',
            (5, 64): 'D',
        }


    @cached_property
    def IMAGE_COMPRESSIONS(self) -> set[int]:
        # set of compression to encode/decode images
        # encode/decode preserves shape and dtype
        # cannot be used with predictors or fillorder
        return {
            6,  # jpeg
            7,  # jpeg
            22610,  # jpegxr
            33003,  # jpeg2k
            33004,  # jpeg2k
            33005,  # jpeg2k
            33007,  # alt_jpeg
            34712,  # jpeg2k
            34892,  # jpeg
            34933,  # png
            34934,  # jpegxr ZIF
            48124,  # jetraw
            50001,  # webp
            50002,  # jpegxl
            52546,  # jpegxl DNG
        }

    @cached_property
    def AXES_NAMES(self) -> dict[str, str]:
        """Map axes character codes to dimension names.

        - **X : width** (image width)
        - **Y : height** (image length)
        - **Z : depth** (image depth)
        - **S : sample** (color space and extra samples)
        - **I : sequence** (generic sequence of images, frames, planes, pages)
        - **T : time** (time series)
        - **C : channel** (acquisition path or emission wavelength)
        - **A : angle** (OME)
        - **P : phase** (OME. In LSM, **P** maps to **position**)
        - **R : tile** (OME. Region, position, or mosaic)
        - **H : lifetime** (OME. Histogram)
        - **E : lambda** (OME. Excitation wavelength)
        - **Q : other** (OME)
        - **L : exposure** (FluoView)
        - **V : event** (FluoView)
        - **M : mosaic** (LSM 6)
        - **J : column** (NDTiff)
        - **K : row** (NDTiff)

        There is no universal standard for dimension codes or names.
        This mapping mainly follows TIFF, OME-TIFF, ImageJ, LSM, and FluoView
        conventions.

        """
        return {
            'X': 'width',
            'Y': 'height',
            'Z': 'depth',
            'S': 'sample',
            'I': 'sequence',
            # 'F': 'file',
            'T': 'time',
            'C': 'channel',
            'A': 'angle',
            'P': 'phase',
            'R': 'tile',
            'H': 'lifetime',
            'E': 'lambda',
            'L': 'exposure',
            'V': 'event',
            'M': 'mosaic',
            'Q': 'other',
            'J': 'column',
            'K': 'row',
        }

    @cached_property
    def AXES_CODES(self) -> dict[str, str]:
        """Map dimension names to axes character codes.

        Reverse mapping of :py:attr:`AXES_NAMES`.

        """
        codes = {name: code for code, name in TIFF.AXES_NAMES.items()}
        codes['z'] = 'Z'  # NDTiff
        codes['position'] = 'R'  # NDTiff
        return codes

    @cached_property
    def GEO_KEYS(self) -> type[enum.IntEnum]:
        """:py:class:`geodb.GeoKeys`."""
        try:
            from .geodb import GeoKeys
        except ImportError:

            class GeoKeys(enum.IntEnum):  # type: ignore[no-redef]
                pass

        return GeoKeys

    @cached_property
    def GEO_CODES(self) -> dict[int, type[enum.IntEnum]]:
        """Map :py:class:`geodb.GeoKeys` to GeoTIFF codes."""
        try:
            from .geodb import GEO_CODES
        except ImportError:
            GEO_CODES = {}
        return GEO_CODES

    '''
    @cached_property
    def PAGE_FLAGS(self) -> set[str]:
        # TiffFile and TiffPage 'is_\*' attributes
        exclude = {
            'reduced',
            'mask',
            'final',
            'memmappable',
            'contiguous',
            'tiled',
            'subsampled',
            'jfif',
        }
        return {
            a[3:]
            for a in dir(TiffPage)
            if a[:3] == 'is_' and a[3:] not in exclude
        }
    '''

    '''
    @cached_property
    def FILE_FLAGS(self) -> set[str]:
        # TiffFile 'is_\*' attributes
        exclude = {'bigtiff', 'appendable'}
        return {
            a[3:]
            for a in dir(TiffFile)
            if a[:3] == 'is_' and a[3:] not in exclude
        }.union(TIFF.PAGE_FLAGS)
    '''

    @property
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

    @property
    def FILE_EXTENSIONS(self) -> tuple[str, ...]:
        """Known TIFF file extensions."""
        return (
            'tif',
            'tiff',
            'ome.tif',
            'lsm',
            'stk',
            'qpi',
            'pcoraw',
            'qptiff',
            'ptiff',
            'ptif',
            'gel',
            'seq',
            'svs',
            'avs',
            'scn',
            'zif',
            'ndpi',
            'bif',
            'tf8',
            'tf2',
            'btf',
            'eer',
        )

    @property
    def FILEOPEN_FILTER(self) -> list[tuple[str, str]]:
        # string for use in Windows File Open box
        return [
            (f'{ext.upper()} files', f'*.{ext}')
            for ext in TIFF.FILE_EXTENSIONS
        ] + [('All files', '*')]

    @property
    def CZ_LSMINFO(self) -> list[tuple[str, str]]:
        # numpy data type of LSMINFO structure
        return [
            ('MagicNumber', 'u4'),
            ('StructureSize', 'i4'),
            ('DimensionX', 'i4'),
            ('DimensionY', 'i4'),
            ('DimensionZ', 'i4'),
            ('DimensionChannels', 'i4'),
            ('DimensionTime', 'i4'),
            ('DataType', 'i4'),  # DATATYPES
            ('ThumbnailX', 'i4'),
            ('ThumbnailY', 'i4'),
            ('VoxelSizeX', 'f8'),
            ('VoxelSizeY', 'f8'),
            ('VoxelSizeZ', 'f8'),
            ('OriginX', 'f8'),
            ('OriginY', 'f8'),
            ('OriginZ', 'f8'),
            ('ScanType', 'u2'),
            ('SpectralScan', 'u2'),
            ('TypeOfData', 'u4'),  # TYPEOFDATA
            ('OffsetVectorOverlay', 'u4'),
            ('OffsetInputLut', 'u4'),
            ('OffsetOutputLut', 'u4'),
            ('OffsetChannelColors', 'u4'),
            ('TimeIntervall', 'f8'),
            ('OffsetChannelDataTypes', 'u4'),
            ('OffsetScanInformation', 'u4'),  # SCANINFO
            ('OffsetKsData', 'u4'),
            ('OffsetTimeStamps', 'u4'),
            ('OffsetEventList', 'u4'),
            ('OffsetRoi', 'u4'),
            ('OffsetBleachRoi', 'u4'),
            ('OffsetNextRecording', 'u4'),
            # LSM 2.0 ends here
            ('DisplayAspectX', 'f8'),
            ('DisplayAspectY', 'f8'),
            ('DisplayAspectZ', 'f8'),
            ('DisplayAspectTime', 'f8'),
            ('OffsetMeanOfRoisOverlay', 'u4'),
            ('OffsetTopoIsolineOverlay', 'u4'),
            ('OffsetTopoProfileOverlay', 'u4'),
            ('OffsetLinescanOverlay', 'u4'),
            ('ToolbarFlags', 'u4'),
            ('OffsetChannelWavelength', 'u4'),
            ('OffsetChannelFactors', 'u4'),
            ('ObjectiveSphereCorrection', 'f8'),
            ('OffsetUnmixParameters', 'u4'),
            # LSM 3.2, 4.0 end here
            ('OffsetAcquisitionParameters', 'u4'),
            ('OffsetCharacteristics', 'u4'),
            ('OffsetPalette', 'u4'),
            ('TimeDifferenceX', 'f8'),
            ('TimeDifferenceY', 'f8'),
            ('TimeDifferenceZ', 'f8'),
            ('InternalUse1', 'u4'),
            ('DimensionP', 'i4'),
            ('DimensionM', 'i4'),
            ('DimensionsReserved', '16i4'),
            ('OffsetTilePositions', 'u4'),
            ('', '9u4'),  # Reserved
            ('OffsetPositions', 'u4'),
            # ('', '21u4'),  # must be 0
        ]

    '''
    @property
    def CZ_LSMINFO_READERS(
        self,
    ) -> dict[str, Callable[[FileHandle], Any] | None]:
        # import functions for CZ_LSMINFO sub-records
        # TODO: read more CZ_LSMINFO sub-records
        return {
            'ScanInformation': read_lsm_scaninfo,
            'TimeStamps': read_lsm_timestamps,
            'EventList': read_lsm_eventlist,
            'ChannelColors': read_lsm_channelcolors,
            'Positions': read_lsm_positions,
            'TilePositions': read_lsm_positions,
            'VectorOverlay': None,
            'InputLut': read_lsm_lookuptable,
            'OutputLut': read_lsm_lookuptable,
            'TimeIntervall': None,
            'ChannelDataTypes': read_lsm_channeldatatypes,
            'KsData': None,
            'Roi': None,
            'BleachRoi': None,
            'NextRecording': None,  # read with TiffFile(fh, offset=)
            'MeanOfRoisOverlay': None,
            'TopoIsolineOverlay': None,
            'TopoProfileOverlay': None,
            'ChannelWavelength': read_lsm_channelwavelength,
            'SphereCorrection': None,
            'ChannelFactors': None,
            'UnmixParameters': None,
            'AcquisitionParameters': None,
            'Characteristics': None,
        }
    '''

    @property
    def CZ_LSMINFO_SCANTYPE(self) -> dict[int, str]:
        # map CZ_LSMINFO.ScanType to dimension order
        return {
            0: 'ZCYX',  # Stack, normal x-y-z-scan
            1: 'CZX',  # Z-Scan, x-z-plane
            2: 'CTX',  # Line or Time Series Line
            3: 'TCYX',  # Time Series Plane, x-y
            4: 'TCZX',  # Time Series z-Scan, x-z
            5: 'CTX',  # Time Series Mean-of-ROIs
            6: 'TZCYX',  # Time Series Stack, x-y-z
            7: 'TZCYX',  # TODO: Spline Scan
            8: 'CZX',  # Spline Plane, x-z
            9: 'TCZX',  # Time Series Spline Plane, x-z
            10: 'CTX',  # Point or Time Series Point
        }

    @property
    def CZ_LSMINFO_DIMENSIONS(self) -> dict[str, str]:
        # map dimension codes to CZ_LSMINFO attribute
        return {
            'X': 'DimensionX',
            'Y': 'DimensionY',
            'Z': 'DimensionZ',
            'C': 'DimensionChannels',
            'T': 'DimensionTime',
            'P': 'DimensionP',
            'M': 'DimensionM',
        }

    @property
    def CZ_LSMINFO_DATATYPES(self) -> dict[int, str]:
        # description of CZ_LSMINFO.DataType
        return {
            0: 'varying data types',
            1: '8 bit unsigned integer',
            2: '12 bit unsigned integer',
            5: '32 bit float',
        }

    @property
    def CZ_LSMINFO_TYPEOFDATA(self) -> dict[int, str]:
        # description of CZ_LSMINFO.TypeOfData
        return {
            0: 'Original scan data',
            1: 'Calculated data',
            2: '3D reconstruction',
            3: 'Topography height map',
        }

    @property
    def CZ_LSMINFO_SCANINFO_ARRAYS(self) -> dict[int, str]:
        return {
            0x20000000: 'Tracks',
            0x30000000: 'Lasers',
            0x60000000: 'DetectionChannels',
            0x80000000: 'IlluminationChannels',
            0xA0000000: 'BeamSplitters',
            0xC0000000: 'DataChannels',
            0x11000000: 'Timers',
            0x13000000: 'Markers',
        }

    @property
    def CZ_LSMINFO_SCANINFO_STRUCTS(self) -> dict[int, str]:
        return {
            # 0x10000000: 'Recording',
            0x40000000: 'Track',
            0x50000000: 'Laser',
            0x70000000: 'DetectionChannel',
            0x90000000: 'IlluminationChannel',
            0xB0000000: 'BeamSplitter',
            0xD0000000: 'DataChannel',
            0x12000000: 'Timer',
            0x14000000: 'Marker',
        }

    @property
    def CZ_LSMINFO_SCANINFO_ATTRIBUTES(self) -> dict[int, str]:
        return {
            # Recording
            0x10000001: 'Name',
            0x10000002: 'Description',
            0x10000003: 'Notes',
            0x10000004: 'Objective',
            0x10000005: 'ProcessingSummary',
            0x10000006: 'SpecialScanMode',
            0x10000007: 'ScanType',
            0x10000008: 'ScanMode',
            0x10000009: 'NumberOfStacks',
            0x1000000A: 'LinesPerPlane',
            0x1000000B: 'SamplesPerLine',
            0x1000000C: 'PlanesPerVolume',
            0x1000000D: 'ImagesWidth',
            0x1000000E: 'ImagesHeight',
            0x1000000F: 'ImagesNumberPlanes',
            0x10000010: 'ImagesNumberStacks',
            0x10000011: 'ImagesNumberChannels',
            0x10000012: 'LinscanXySize',
            0x10000013: 'ScanDirection',
            0x10000014: 'TimeSeries',
            0x10000015: 'OriginalScanData',
            0x10000016: 'ZoomX',
            0x10000017: 'ZoomY',
            0x10000018: 'ZoomZ',
            0x10000019: 'Sample0X',
            0x1000001A: 'Sample0Y',
            0x1000001B: 'Sample0Z',
            0x1000001C: 'SampleSpacing',
            0x1000001D: 'LineSpacing',
            0x1000001E: 'PlaneSpacing',
            0x1000001F: 'PlaneWidth',
            0x10000020: 'PlaneHeight',
            0x10000021: 'VolumeDepth',
            0x10000023: 'Nutation',
            0x10000034: 'Rotation',
            0x10000035: 'Precession',
            0x10000036: 'Sample0time',
            0x10000037: 'StartScanTriggerIn',
            0x10000038: 'StartScanTriggerOut',
            0x10000039: 'StartScanEvent',
            0x10000040: 'StartScanTime',
            0x10000041: 'StopScanTriggerIn',
            0x10000042: 'StopScanTriggerOut',
            0x10000043: 'StopScanEvent',
            0x10000044: 'StopScanTime',
            0x10000045: 'UseRois',
            0x10000046: 'UseReducedMemoryRois',
            0x10000047: 'User',
            0x10000048: 'UseBcCorrection',
            0x10000049: 'PositionBcCorrection1',
            0x10000050: 'PositionBcCorrection2',
            0x10000051: 'InterpolationY',
            0x10000052: 'CameraBinning',
            0x10000053: 'CameraSupersampling',
            0x10000054: 'CameraFrameWidth',
            0x10000055: 'CameraFrameHeight',
            0x10000056: 'CameraOffsetX',
            0x10000057: 'CameraOffsetY',
            0x10000059: 'RtBinning',
            0x1000005A: 'RtFrameWidth',
            0x1000005B: 'RtFrameHeight',
            0x1000005C: 'RtRegionWidth',
            0x1000005D: 'RtRegionHeight',
            0x1000005E: 'RtOffsetX',
            0x1000005F: 'RtOffsetY',
            0x10000060: 'RtZoom',
            0x10000061: 'RtLinePeriod',
            0x10000062: 'Prescan',
            0x10000063: 'ScanDirectionZ',
            # Track
            0x40000001: 'MultiplexType',  # 0 After Line; 1 After Frame
            0x40000002: 'MultiplexOrder',
            0x40000003: 'SamplingMode',  # 0 Sample; 1 Line Avg; 2 Frame Avg
            0x40000004: 'SamplingMethod',  # 1 Mean; 2 Sum
            0x40000005: 'SamplingNumber',
            0x40000006: 'Acquire',
            0x40000007: 'SampleObservationTime',
            0x4000000B: 'TimeBetweenStacks',
            0x4000000C: 'Name',
            0x4000000D: 'Collimator1Name',
            0x4000000E: 'Collimator1Position',
            0x4000000F: 'Collimator2Name',
            0x40000010: 'Collimator2Position',
            0x40000011: 'IsBleachTrack',
            0x40000012: 'IsBleachAfterScanNumber',
            0x40000013: 'BleachScanNumber',
            0x40000014: 'TriggerIn',
            0x40000015: 'TriggerOut',
            0x40000016: 'IsRatioTrack',
            0x40000017: 'BleachCount',
            0x40000018: 'SpiCenterWavelength',
            0x40000019: 'PixelTime',
            0x40000021: 'CondensorFrontlens',
            0x40000023: 'FieldStopValue',
            0x40000024: 'IdCondensorAperture',
            0x40000025: 'CondensorAperture',
            0x40000026: 'IdCondensorRevolver',
            0x40000027: 'CondensorFilter',
            0x40000028: 'IdTransmissionFilter1',
            0x40000029: 'IdTransmission1',
            0x40000030: 'IdTransmissionFilter2',
            0x40000031: 'IdTransmission2',
            0x40000032: 'RepeatBleach',
            0x40000033: 'EnableSpotBleachPos',
            0x40000034: 'SpotBleachPosx',
            0x40000035: 'SpotBleachPosy',
            0x40000036: 'SpotBleachPosz',
            0x40000037: 'IdTubelens',
            0x40000038: 'IdTubelensPosition',
            0x40000039: 'TransmittedLight',
            0x4000003A: 'ReflectedLight',
            0x4000003B: 'SimultanGrabAndBleach',
            0x4000003C: 'BleachPixelTime',
            # Laser
            0x50000001: 'Name',
            0x50000002: 'Acquire',
            0x50000003: 'Power',
            # DetectionChannel
            0x70000001: 'IntegrationMode',
            0x70000002: 'SpecialMode',
            0x70000003: 'DetectorGainFirst',
            0x70000004: 'DetectorGainLast',
            0x70000005: 'AmplifierGainFirst',
            0x70000006: 'AmplifierGainLast',
            0x70000007: 'AmplifierOffsFirst',
            0x70000008: 'AmplifierOffsLast',
            0x70000009: 'PinholeDiameter',
            0x7000000A: 'CountingTrigger',
            0x7000000B: 'Acquire',
            0x7000000C: 'PointDetectorName',
            0x7000000D: 'AmplifierName',
            0x7000000E: 'PinholeName',
            0x7000000F: 'FilterSetName',
            0x70000010: 'FilterName',
            0x70000013: 'IntegratorName',
            0x70000014: 'ChannelName',
            0x70000015: 'DetectorGainBc1',
            0x70000016: 'DetectorGainBc2',
            0x70000017: 'AmplifierGainBc1',
            0x70000018: 'AmplifierGainBc2',
            0x70000019: 'AmplifierOffsetBc1',
            0x70000020: 'AmplifierOffsetBc2',
            0x70000021: 'SpectralScanChannels',
            0x70000022: 'SpiWavelengthStart',
            0x70000023: 'SpiWavelengthStop',
            0x70000026: 'DyeName',
            0x70000027: 'DyeFolder',
            # IlluminationChannel
            0x90000001: 'Name',
            0x90000002: 'Power',
            0x90000003: 'Wavelength',
            0x90000004: 'Aquire',
            0x90000005: 'DetchannelName',
            0x90000006: 'PowerBc1',
            0x90000007: 'PowerBc2',
            # BeamSplitter
            0xB0000001: 'FilterSet',
            0xB0000002: 'Filter',
            0xB0000003: 'Name',
            # DataChannel
            0xD0000001: 'Name',
            0xD0000003: 'Acquire',
            0xD0000004: 'Color',
            0xD0000005: 'SampleType',
            0xD0000006: 'BitsPerSample',
            0xD0000007: 'RatioType',
            0xD0000008: 'RatioTrack1',
            0xD0000009: 'RatioTrack2',
            0xD000000A: 'RatioChannel1',
            0xD000000B: 'RatioChannel2',
            0xD000000C: 'RatioConst1',
            0xD000000D: 'RatioConst2',
            0xD000000E: 'RatioConst3',
            0xD000000F: 'RatioConst4',
            0xD0000010: 'RatioConst5',
            0xD0000011: 'RatioConst6',
            0xD0000012: 'RatioFirstImages1',
            0xD0000013: 'RatioFirstImages2',
            0xD0000014: 'DyeName',
            0xD0000015: 'DyeFolder',
            0xD0000016: 'Spectrum',
            0xD0000017: 'Acquire',
            # Timer
            0x12000001: 'Name',
            0x12000002: 'Description',
            0x12000003: 'Interval',
            0x12000004: 'TriggerIn',
            0x12000005: 'TriggerOut',
            0x12000006: 'ActivationTime',
            0x12000007: 'ActivationNumber',
            # Marker
            0x14000001: 'Name',
            0x14000002: 'Description',
            0x14000003: 'TriggerIn',
            0x14000004: 'TriggerOut',
        }

    @cached_property
    def CZ_LSM_LUTTYPE(self):  # TODO: type this
        class CZ_LSM_LUTTYPE(enum.IntEnum):
            NORMAL = 0
            ORIGINAL = 1
            RAMP = 2
            POLYLINE = 3
            SPLINE = 4
            GAMMA = 5

        return CZ_LSM_LUTTYPE

    @cached_property
    def CZ_LSM_SUBBLOCK_TYPE(self):  # TODO: type this
        class CZ_LSM_SUBBLOCK_TYPE(enum.IntEnum):
            END = 0
            GAMMA = 1
            BRIGHTNESS = 2
            CONTRAST = 3
            RAMP = 4
            KNOTS = 5
            PALETTE_12_TO_12 = 6

        return CZ_LSM_SUBBLOCK_TYPE

    @property
    def NIH_IMAGE_HEADER(self):  # TODO: type this
        return [
            ('FileID', 'S8'),
            ('nLines', 'i2'),
            ('PixelsPerLine', 'i2'),
            ('Version', 'i2'),
            ('OldLutMode', 'i2'),
            ('OldnColors', 'i2'),
            ('Colors', 'u1', (3, 32)),
            ('OldColorStart', 'i2'),
            ('ColorWidth', 'i2'),
            ('ExtraColors', 'u2', (6, 3)),
            ('nExtraColors', 'i2'),
            ('ForegroundIndex', 'i2'),
            ('BackgroundIndex', 'i2'),
            ('XScale', 'f8'),
            ('Unused2', 'i2'),
            ('Unused3', 'i2'),
            ('UnitsID', 'i2'),  # NIH_UNITS_TYPE
            ('p1', [('x', 'i2'), ('y', 'i2')]),
            ('p2', [('x', 'i2'), ('y', 'i2')]),
            ('CurveFitType', 'i2'),  # NIH_CURVEFIT_TYPE
            ('nCoefficients', 'i2'),
            ('Coeff', 'f8', 6),
            ('UMsize', 'u1'),
            ('UM', 'S15'),
            ('UnusedBoolean', 'u1'),
            ('BinaryPic', 'b1'),
            ('SliceStart', 'i2'),
            ('SliceEnd', 'i2'),
            ('ScaleMagnification', 'f4'),
            ('nSlices', 'i2'),
            ('SliceSpacing', 'f4'),
            ('CurrentSlice', 'i2'),
            ('FrameInterval', 'f4'),
            ('PixelAspectRatio', 'f4'),
            ('ColorStart', 'i2'),
            ('ColorEnd', 'i2'),
            ('nColors', 'i2'),
            ('Fill1', '3u2'),
            ('Fill2', '3u2'),
            ('Table', 'u1'),  # NIH_COLORTABLE_TYPE
            ('LutMode', 'u1'),  # NIH_LUTMODE_TYPE
            ('InvertedTable', 'b1'),
            ('ZeroClip', 'b1'),
            ('XUnitSize', 'u1'),
            ('XUnit', 'S11'),
            ('StackType', 'i2'),  # NIH_STACKTYPE_TYPE
            # ('UnusedBytes', 'u1', 200)
        ]

    @property
    def NIH_COLORTABLE_TYPE(self) -> tuple[str, ...]:
        return (
            'CustomTable',
            'AppleDefault',
            'Pseudo20',
            'Pseudo32',
            'Rainbow',
            'Fire1',
            'Fire2',
            'Ice',
            'Grays',
            'Spectrum',
        )

    @property
    def NIH_LUTMODE_TYPE(self) -> tuple[str, ...]:
        return (
            'PseudoColor',
            'OldAppleDefault',
            'OldSpectrum',
            'GrayScale',
            'ColorLut',
            'CustomGrayscale',
        )

    @property
    def NIH_CURVEFIT_TYPE(self) -> tuple[str, ...]:
        return (
            'StraightLine',
            'Poly2',
            'Poly3',
            'Poly4',
            'Poly5',
            'ExpoFit',
            'PowerFit',
            'LogFit',
            'RodbardFit',
            'SpareFit1',
            'Uncalibrated',
            'UncalibratedOD',
        )

    @property
    def NIH_UNITS_TYPE(self) -> tuple[str, ...]:
        return (
            'Nanometers',
            'Micrometers',
            'Millimeters',
            'Centimeters',
            'Meters',
            'Kilometers',
            'Inches',
            'Feet',
            'Miles',
            'Pixels',
            'OtherUnits',
        )

    @property
    def TVIPS_HEADER_V1(self) -> list[tuple[str, str]]:
        # TVIPS TemData structure from EMMENU Help file
        return [
            ('Version', 'i4'),
            ('CommentV1', 'S80'),
            ('HighTension', 'i4'),
            ('SphericalAberration', 'i4'),
            ('IlluminationAperture', 'i4'),
            ('Magnification', 'i4'),
            ('PostMagnification', 'i4'),
            ('FocalLength', 'i4'),
            ('Defocus', 'i4'),
            ('Astigmatism', 'i4'),
            ('AstigmatismDirection', 'i4'),
            ('BiprismVoltage', 'i4'),
            ('SpecimenTiltAngle', 'i4'),
            ('SpecimenTiltDirection', 'i4'),
            ('IlluminationTiltDirection', 'i4'),
            ('IlluminationTiltAngle', 'i4'),
            ('ImageMode', 'i4'),
            ('EnergySpread', 'i4'),
            ('ChromaticAberration', 'i4'),
            ('ShutterType', 'i4'),
            ('DefocusSpread', 'i4'),
            ('CcdNumber', 'i4'),
            ('CcdSize', 'i4'),
            ('OffsetXV1', 'i4'),
            ('OffsetYV1', 'i4'),
            ('PhysicalPixelSize', 'i4'),
            ('Binning', 'i4'),
            ('ReadoutSpeed', 'i4'),
            ('GainV1', 'i4'),
            ('SensitivityV1', 'i4'),
            ('ExposureTimeV1', 'i4'),
            ('FlatCorrected', 'i4'),
            ('DeadPxCorrected', 'i4'),
            ('ImageMean', 'i4'),
            ('ImageStd', 'i4'),
            ('DisplacementX', 'i4'),
            ('DisplacementY', 'i4'),
            ('DateV1', 'i4'),
            ('TimeV1', 'i4'),
            ('ImageMin', 'i4'),
            ('ImageMax', 'i4'),
            ('ImageStatisticsQuality', 'i4'),
        ]

    @property
    def TVIPS_HEADER_V2(self) -> list[tuple[str, str]]:
        return [
            ('ImageName', 'V160'),  # utf16
            ('ImageFolder', 'V160'),
            ('ImageSizeX', 'i4'),
            ('ImageSizeY', 'i4'),
            ('ImageSizeZ', 'i4'),
            ('ImageSizeE', 'i4'),
            ('ImageDataType', 'i4'),
            ('Date', 'i4'),
            ('Time', 'i4'),
            ('Comment', 'V1024'),
            ('ImageHistory', 'V1024'),
            ('Scaling', '16f4'),
            ('ImageStatistics', '16c16'),
            ('ImageType', 'i4'),
            ('ImageDisplaType', 'i4'),
            ('PixelSizeX', 'f4'),  # distance between two px in x, [nm]
            ('PixelSizeY', 'f4'),  # distance between two px in y, [nm]
            ('ImageDistanceZ', 'f4'),
            ('ImageDistanceE', 'f4'),
            ('ImageMisc', '32f4'),
            ('TemType', 'V160'),
            ('TemHighTension', 'f4'),
            ('TemAberrations', '32f4'),
            ('TemEnergy', '32f4'),
            ('TemMode', 'i4'),
            ('TemMagnification', 'f4'),
            ('TemMagnificationCorrection', 'f4'),
            ('PostMagnification', 'f4'),
            ('TemStageType', 'i4'),
            ('TemStagePosition', '5f4'),  # x, y, z, a, b
            ('TemImageShift', '2f4'),
            ('TemBeamShift', '2f4'),
            ('TemBeamTilt', '2f4'),
            ('TilingParameters', '7f4'),  # 0: tiling? 1:x 2:y 3: max x
            #                               4: max y 5: overlap x 6: overlap y
            ('TemIllumination', '3f4'),  # 0: spotsize 1: intensity
            ('TemShutter', 'i4'),
            ('TemMisc', '32f4'),
            ('CameraType', 'V160'),
            ('PhysicalPixelSizeX', 'f4'),
            ('PhysicalPixelSizeY', 'f4'),
            ('OffsetX', 'i4'),
            ('OffsetY', 'i4'),
            ('BinningX', 'i4'),
            ('BinningY', 'i4'),
            ('ExposureTime', 'f4'),
            ('Gain', 'f4'),
            ('ReadoutRate', 'f4'),
            ('FlatfieldDescription', 'V160'),
            ('Sensitivity', 'f4'),
            ('Dose', 'f4'),
            ('CamMisc', '32f4'),
            ('FeiMicroscopeInformation', 'V1024'),
            ('FeiSpecimenInformation', 'V1024'),
            ('Magic', 'u4'),
        ]

    @property
    def MM_HEADER(self) -> list[tuple[Any, ...]]:
        # Olympus FluoView MM_Header
        MM_DIMENSION = [
            ('Name', 'S16'),
            ('Size', 'i4'),
            ('Origin', 'f8'),
            ('Resolution', 'f8'),
            ('Unit', 'S64'),
        ]
        return [
            ('HeaderFlag', 'i2'),
            ('ImageType', 'u1'),
            ('ImageName', 'S257'),
            ('OffsetData', 'u4'),
            ('PaletteSize', 'i4'),
            ('OffsetPalette0', 'u4'),
            ('OffsetPalette1', 'u4'),
            ('CommentSize', 'i4'),
            ('OffsetComment', 'u4'),
            ('Dimensions', MM_DIMENSION, 10),
            ('OffsetPosition', 'u4'),
            ('MapType', 'i2'),
            ('MapMin', 'f8'),
            ('MapMax', 'f8'),
            ('MinValue', 'f8'),
            ('MaxValue', 'f8'),
            ('OffsetMap', 'u4'),
            ('Gamma', 'f8'),
            ('Offset', 'f8'),
            ('GrayChannel', MM_DIMENSION),
            ('OffsetThumbnail', 'u4'),
            ('VoiceField', 'i4'),
            ('OffsetVoiceField', 'u4'),
        ]

    @property
    def MM_DIMENSIONS(self) -> dict[str, str]:
        # map FluoView MM_Header.Dimensions to axes characters
        return {
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'T': 'T',
            'CH': 'C',
            'WAVELENGTH': 'C',
            'TIME': 'T',
            'XY': 'R',
            'EVENT': 'V',
            'EXPOSURE': 'L',
        }

    '''
    @property
    def UIC_TAGS(self) -> list[tuple[str, Any]]:
        # map Universal Imaging Corporation MetaMorph internal tag ids to
        # name and type
        from fractions import Fraction

        return [
            ('AutoScale', int),
            ('MinScale', int),
            ('MaxScale', int),
            ('SpatialCalibration', int),
            ('XCalibration', Fraction),
            ('YCalibration', Fraction),
            ('CalibrationUnits', str),
            ('Name', str),
            ('ThreshState', int),
            ('ThreshStateRed', int),
            ('tagid_10', None),  # undefined
            ('ThreshStateGreen', int),
            ('ThreshStateBlue', int),
            ('ThreshStateLo', int),
            ('ThreshStateHi', int),
            ('Zoom', int),
            ('CreateTime', julian_datetime),
            ('LastSavedTime', julian_datetime),
            ('currentBuffer', int),
            ('grayFit', None),
            ('grayPointCount', None),
            ('grayX', Fraction),
            ('grayY', Fraction),
            ('grayMin', Fraction),
            ('grayMax', Fraction),
            ('grayUnitName', str),
            ('StandardLUT', int),
            ('wavelength', int),
            ('StagePosition', '(%i,2,2)u4'),  # N xy positions as fract
            ('CameraChipOffset', '(%i,2,2)u4'),  # N xy offsets as fract
            ('OverlayMask', None),
            ('OverlayCompress', None),
            ('Overlay', None),
            ('SpecialOverlayMask', None),
            ('SpecialOverlayCompress', None),
            ('SpecialOverlay', None),
            ('ImageProperty', read_uic_property),
            ('StageLabel', '%ip'),  # N str
            ('AutoScaleLoInfo', Fraction),
            ('AutoScaleHiInfo', Fraction),
            ('AbsoluteZ', '(%i,2)u4'),  # N fractions
            ('AbsoluteZValid', '(%i,)u4'),  # N long
            ('Gamma', 'I'),  # 'I' uses offset
            ('GammaRed', 'I'),
            ('GammaGreen', 'I'),
            ('GammaBlue', 'I'),
            ('CameraBin', '2I'),
            ('NewLUT', int),
            ('ImagePropertyEx', None),
            ('PlaneProperty', int),
            ('UserLutTable', '(256,3)u1'),
            ('RedAutoScaleInfo', int),
            ('RedAutoScaleLoInfo', Fraction),
            ('RedAutoScaleHiInfo', Fraction),
            ('RedMinScaleInfo', int),
            ('RedMaxScaleInfo', int),
            ('GreenAutoScaleInfo', int),
            ('GreenAutoScaleLoInfo', Fraction),
            ('GreenAutoScaleHiInfo', Fraction),
            ('GreenMinScaleInfo', int),
            ('GreenMaxScaleInfo', int),
            ('BlueAutoScaleInfo', int),
            ('BlueAutoScaleLoInfo', Fraction),
            ('BlueAutoScaleHiInfo', Fraction),
            ('BlueMinScaleInfo', int),
            ('BlueMaxScaleInfo', int),
            # ('OverlayPlaneColor', read_uic_overlay_plane_color),
        ]
    '''

    @property
    def PILATUS_HEADER(self) -> dict[str, Any]:
        # PILATUS CBF Header Specification, Version 1.4
        # map key to [value_indices], type
        return {
            'Detector': ([slice(1, None)], str),
            'Pixel_size': ([1, 4], float),
            'Silicon': ([3], float),
            'Exposure_time': ([1], float),
            'Exposure_period': ([1], float),
            'Tau': ([1], float),
            'Count_cutoff': ([1], int),
            'Threshold_setting': ([1], float),
            'Gain_setting': ([1, 2], str),
            'N_excluded_pixels': ([1], int),
            'Excluded_pixels': ([1], str),
            'Flat_field': ([1], str),
            'Trim_file': ([1], str),
            'Image_path': ([1], str),
            # optional
            'Wavelength': ([1], float),
            'Energy_range': ([1, 2], float),
            'Detector_distance': ([1], float),
            'Detector_Voffset': ([1], float),
            'Beam_xy': ([1, 2], float),
            'Flux': ([1], str),
            'Filter_transmission': ([1], float),
            'Start_angle': ([1], float),
            'Angle_increment': ([1], float),
            'Detector_2theta': ([1], float),
            'Polarization': ([1], float),
            'Alpha': ([1], float),
            'Kappa': ([1], float),
            'Phi': ([1], float),
            'Phi_increment': ([1], float),
            'Chi': ([1], float),
            'Chi_increment': ([1], float),
            'Oscillation_axis': ([slice(1, None)], str),
            'N_oscillations': ([1], int),
            'Start_position': ([1], float),
            'Position_increment': ([1], float),
            'Shutter_time': ([1], float),
            'Omega': ([1], float),
            'Omega_increment': ([1], float),
        }

    @cached_property
    def ALLOCATIONGRANULARITY(self) -> int:
        # alignment for writing contiguous data to TIFF
        import mmap

        return mmap.ALLOCATIONGRANULARITY

    @cached_property
    def MAXWORKERS(self) -> int:
        """Default maximum number of threads for de/compressing segments.

        The value of the ``TIFFFILE_NUM_THREADS`` environment variable if set,
        else half the CPU cores up to 32.

        """
        if 'TIFFFILE_NUM_THREADS' in os.environ:
            return max(1, int(os.environ['TIFFFILE_NUM_THREADS']))
        cpu_count: int | None
        try:
            cpu_count = len(
                os.sched_getaffinity(0)  # type: ignore[attr-defined]
            )
        except AttributeError:
            cpu_count = os.cpu_count()
        if cpu_count is None:
            return 1
        return min(32, max(1, cpu_count // 2))

    @cached_property
    def MAXIOWORKERS(self) -> int:
        """Default maximum number of I/O threads for reading file sequences.

        The value of the ``TIFFFILE_NUM_IOTHREADS`` environment variable if
        set, else 4 more than the number of CPU cores up to 32.

        """
        if 'TIFFFILE_NUM_IOTHREADS' in os.environ:
            return max(1, int(os.environ['TIFFFILE_NUM_IOTHREADS']))
        cpu_count: int | None
        try:
            cpu_count = len(
                os.sched_getaffinity(0)  # type: ignore[attr-defined]
            )
        except AttributeError:
            cpu_count = os.cpu_count()
        if cpu_count is None:
            return 5
        return min(32, cpu_count + 4)

    BUFFERSIZE: int = 268435456
    """Default number of bytes to read or encode in one pass (256 MB)."""


TIFF = _TIFF()

class TiffFileError(Exception):
    """Exception to indicate invalid TIFF structure."""