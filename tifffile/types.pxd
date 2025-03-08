
cpdef enum class DATATYPE:
    """TIFF tag data types."""

    BYTE = 1
    # 8-bit unsigned integer."""
    ASCII = 2
    # 8-bit byte with last byte null, containing 7-bit ASCII code."""
    SHORT = 3
    # 16-bit unsigned integer."""
    LONG = 4
    # 32-bit unsigned integer."""
    RATIONAL = 5
    # Two 32-bit unsigned integers, numerator and denominator of fraction."""
    SBYTE = 6
    # 8-bit signed integer."""
    UNDEFINED = 7
    # 8-bit byte that may contain anything."""
    SSHORT = 8
    # 16-bit signed integer."""
    SLONG = 9
    # 32-bit signed integer."""
    SRATIONAL = 10
    # Two 32-bit signed integers, numerator and denominator of fraction."""
    FLOAT = 11
    # Single precision (4-byte) IEEE format."""
    DOUBLE = 12
    # Double precision (8-byte) IEEE format."""
    IFD = 13
    # Unsigned 4 byte IFD offset."""
    UNICODE = 14
    # UTF-16 (2-byte) unicode string."""
    COMPLEX = 15
    # Single precision (8-byte) complex number."""
    LONG8 = 16
    # Unsigned 8 byte integer (BigTIFF)."""
    SLONG8 = 17
    # Signed 8 byte integer (BigTIFF)."""
    IFD8 = 18
    # Unsigned 8 byte IFD offset (BigTIFF)."""


cpdef enum class COMPRESSION:
    """Values of Compression tag.

    Compression scheme used on image data.

    """

    NONE = 1
    # No compression (default)."""
    CCITTRLE = 2  # CCITT 1D
    CCITT_T4 = 3  # T4/Group 3 Fax
    CCITT_T6 = 4  # T6/Group 4 Fax
    LZW = 5
    # Lempel-Ziv-Welch."""
    OJPEG = 6  # old-style JPEG
    JPEG = 7
    # New style JPEG."""
    ADOBE_DEFLATE = 8
    # Deflate, aka ZLIB."""
    JBIG_BW = 9  # VC5
    JBIG_COLOR = 10
    JPEG_99 = 99
    KODAK_262 = 262
    JPEGXR_NDPI = 22610
    # JPEG XR (Hammatsu NDPI)."""
    NEXT = 32766
    SONY_ARW = 32767
    PACKED_RAW = 32769
    SAMSUNG_SRW = 32770
    CCIRLEW = 32771  # Word-aligned 1D Huffman compression
    SAMSUNG_SRW2 = 32772
    PACKBITS = 32773
    # PackBits, aka Macintosh RLE."""
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
    # JPEG 2000 YCbCr (Leica Aperio)."""
    JPEG_2000_LOSSY = 33004
    # Lossy JPEG 2000 (Bio-Formats)."""
    APERIO_JP2000_RGB = 33005  # Kakadu libraries
    # JPEG 2000 RGB (Leica Aperio)."""
    ALT_JPEG = 33007
    # JPEG (Bio-Formats)."""
    # PANASONIC_RAW1 = 34316
    # PANASONIC_RAW2 = 34826
    # PANASONIC_RAW3 = 34828
    # PANASONIC_RAW4 = 34830
    JBIG = 34661
    SGILOG = 34676  # LogLuv32
    SGILOG24 = 34677
    JPEG2000 = 34712
    # JPEG 2000."""
    NIKON_NEF = 34713
    JBIG2 = 34715
    MDI_BINARY = 34718  # Microsoft Document Imaging
    MDI_PROGRESSIVE = 34719  # Microsoft Document Imaging
    MDI_VECTOR = 34720  # Microsoft Document Imaging
    LERC = 34887
    # ESRI Limited Error Raster Compression."""
    JPEG_LOSSY = 34892  # DNG
    LZMA = 34925
    # Lempel-Ziv-Markov chain Algorithm."""
    ZSTD_DEPRECATED = 34926
    WEBP_DEPRECATED = 34927
    PNG = 34933  # Objective Pathology Services
    # Portable Network Graphics (Zoomable Image File format)."""
    JPEGXR = 34934
    # JPEG XR (Zoomable Image File format)."""
    JETRAW = 48124
    # Jetraw by Dotphoton."""
    ZSTD = 50000
    # Zstandard."""
    WEBP = 50001
    # WebP."""
    JPEGXL = 50002  # GDAL
    # JPEG XL."""
    PIXTIFF = 50013
    # ZLIB (Atalasoft)."""
    JPEGXL_DNG = 52546
    # JPEG XL (DNG)."""
    EER_V0 = 65000  # FIXED82 Thermo Fisher Scientific
    EER_V1 = 65001  # FIXED72 Thermo Fisher Scientific
    EER_V2 = 65002  # VARIABLE Thermo Fisher Scientific
    # KODAK_DCR = 65000
    # PENTAX_PEF = 65535

    #def __bool__(self) -> bool:
    #    return self > 1


cpdef enum class PREDICTOR:
    """Values of Predictor tag.

    A mathematical operator that is applied to the image data before
    compression.

    """

    NONE = 1
    # No prediction scheme used (default)."""
    HORIZONTAL = 2
    # Horizontal differencing."""
    FLOATINGPOINT = 3
    # Floating-point horizontal differencing."""
    HORIZONTALX2 = 34892  # DNG
    HORIZONTALX4 = 34893
    FLOATINGPOINTX2 = 34894
    FLOATINGPOINTX4 = 34895

    #def __bool__(self) -> bool:
    #    return self > 1


cpdef enum class PHOTOMETRIC:
    """Values of PhotometricInterpretation tag.

    The color space of the image.

    """

    MINISWHITE = 0
    # For bilevel and grayscale images, 0 is imaged as white."""
    MINISBLACK = 1
    # For bilevel and grayscale images, 0 is imaged as black."""
    RGB = 2
    # Chroma components are Red, Green, Blue."""
    PALETTE = 3
    # Single chroma component is index into colormap."""
    MASK = 4
    SEPARATED = 5
    # Chroma components are Cyan, Magenta, Yellow, and Key (black)."""
    YCBCR = 6
    # Chroma components are Luma, blue-difference, and red-difference."""
    CIELAB = 8
    ICCLAB = 9
    ITULAB = 10
    CFA = 32803
    # Color Filter Array."""
    LOGL = 32844
    LOGLUV = 32845
    LINEAR_RAW = 34892
    DEPTH_MAP = 51177  # DNG 1.5
    SEMANTIC_MASK = 52527  # DNG 1.6


cpdef enum class FILETYPE:
    """Values of NewSubfileType tag.

    A general indication of the kind of the image.

    """

    UNDEFINED = 0
    # Image is full-resolution (default)."""
    REDUCEDIMAGE = 1
    # Image is reduced-resolution version of another image."""
    PAGE = 2
    # Image is single page of multi-page image."""
    MASK = 4
    # Image is transparency mask for another image."""
    MACRO = 8  # Aperio SVS, or DNG Depth map
    # Image is MACRO image (SVS) or depth map for another image (DNG)."""
    ENHANCED = 16  # DNG
    # Image contains enhanced image (DNG)."""
    DNG = 65536  # 65537: Alternative, 65540: Semantic mask


cpdef enum class OFILETYPE:
    """Values of deprecated SubfileType tag."""

    UNDEFINED = 0
    IMAGE = 1  # full-resolution image
    REDUCEDIMAGE = 2  # reduced-resolution image
    PAGE = 3  # single page of multi-page image


cpdef enum class FILLORDER:
    """Values of FillOrder tag.

    The logical order of bits within a byte.

    """

    MSB2LSB = 1
    # Pixel values are stored in higher-order bits of byte (default)."""
    LSB2MSB = 2
    # Pixels values are stored in lower-order bits of byte."""


cpdef enum class ORIENTATION:
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


cpdef enum class PLANARCONFIG:
    """Values of PlanarConfiguration tag.

    Specifies how components of each pixel are stored.

    """

    CONTIG = 1
    # Chunky, component values are stored contiguously (default)."""
    SEPARATE = 2
    # Planar, component values are stored in separate planes."""


cpdef enum class RESUNIT:
    """Values of ResolutionUnit tag.

    The unit of measurement for XResolution and YResolution.

    """

    NONE = 1
    # No absolute unit of measurement."""
    INCH = 2
    # Inch (default)."""
    CENTIMETER = 3
    # Centimeter."""
    MILLIMETER = 4
    # Millimeter (DNG)."""
    MICROMETER = 5
    # Micrometer (DNG)."""

    #def __bool__(self) -> bool:
    #    return self > 1


cpdef enum class EXTRASAMPLE:
    """Values of ExtraSamples tag.

    Interpretation of extra components in a pixel.

    """

    UNSPECIFIED = 0
    # Unspecified data."""
    ASSOCALPHA = 1
    # Associated alpha data with premultiplied color."""
    UNASSALPHA = 2
    # Unassociated alpha data."""


cpdef enum class SAMPLEFORMAT:
    """Values of SampleFormat tag.

    Data type of samples in a pixel.

    """

    UINT = 1
    # Unsigned integer."""
    INT = 2
    # Signed integer."""
    IEEEFP = 3
    # IEEE floating-point"""
    VOID = 4
    # Undefined."""
    COMPLEXINT = 5
    # Complex integer."""
    COMPLEXIEEEFP = 6
    # Complex floating-point."""


cpdef enum class CHUNKMODE:
    """ZarrStore chunk modes.

    Specifies how to chunk data in Zarr 2 stores.

    """

    STRILE = 0
    # Chunk is strip or tile."""
    PLANE = 1
    # Chunk is image plane."""
    PAGE = 2
    # Chunk is image in page."""
    FILE = 3
    # Chunk is image in file."""
