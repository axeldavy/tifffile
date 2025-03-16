#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
from cython.view cimport array
from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, fopen, fclose, fread, fseek, ftell, SEEK_SET, SEEK_END, SEEK_CUR

cimport cython

"""
Implementation of a custom tiff reader
that focuses on the specific case of
TiffFormatClassicLE with zstd compression,
tiling and horizontal predictor, uint16 ot uint8
single channel data.

Optimization principles:
- classes are instanciated once and reused
- similarly buffers are reused as much as possible
- the GIL is not held as much as possible
"""

# ZSTD C API declarations
cdef extern from "zstddeclib.c" nogil:
    ctypedef struct ZSTD_DCtx:
        pass
    
    size_t ZSTD_getFrameContentSize(const void* src, size_t srcSize)
    unsigned long long ZSTD_CONTENTSIZE_UNKNOWN
    unsigned long long ZSTD_CONTENTSIZE_ERROR
    
    ZSTD_DCtx* ZSTD_createDCtx()
    size_t ZSTD_freeDCtx(ZSTD_DCtx* dctx)
    size_t ZSTD_decompressDCtx(ZSTD_DCtx* dctx, void* dst, size_t dstCapacity, const void* src, size_t srcSize)
    size_t ZSTD_decompress(void* dst, size_t dstCapacity, const void* src, size_t srcSize)
    size_t ZSTD_isError(size_t code)
    const char* ZSTD_getErrorName(size_t code)

cdef inline void delta_decode_uint8(
    uint8_t* buffer,
    int64_t width,
    int64_t height,
    int64_t stride) noexcept nogil:
    """
    Decode delta encoded uint8 buffer in place.
    Implements horizontal differencing (TIFF predictor=2).
    (No endianess conversion)
    
    Parameters:
        buffer: Pointer to buffer to decode
        width: Width of the image in pixels
        height: Height of the image in pixels
        stride: Number of elements between rows (can be >= width)
    """
    cdef int64_t x, y
    cdef uint8_t* row
    cdef uint8_t* row_end

    for y in range(height):
        row = buffer + y * stride
        for x in range(1, width):
            row[x] += row[x-1]

cdef inline void delta_decode_uint16(
    uint16_t* buffer,
    int64_t width,
    int64_t height,
    int64_t stride) noexcept nogil:
    """
    Decode delta encoded uint16 buffer in place.
    Implements horizontal differencing (TIFF predictor=2).
    (No endianess conversion)
    
    Parameters:
        buffer: Pointer to buffer to decode
        width: Width of the image in pixels
        height: Height of the image in pixels
        stride: Number of elements between rows (can be >= width)
    """
    cdef int64_t x, y
    cdef uint16_t* row
    cdef uint16_t* row_end

    for y in range(height):
        row = buffer + y * stride
        for x in range(1, width):
            row[x] += row[x-1]


cdef extern from * nogil:
    """
    // Classic TIFF header structure (Little-Endian)
    #pragma pack(push, 1)
    struct TiffHeader {
        // "II" (0x4949) for little-endian
        char byteOrder[2];
        
        // 42 (0x002A) for classic TIFF
        uint16_t version;
        
        // Offset to first IFD
        uint32_t firstIFDOffset;
    };
    #pragma pack(pop)

    #pragma pack(push, 1)
    struct IFDHeader {
        uint16_t numEntries;
        // Tag entries follow (variable number)
        // Next IFD offset (uint32_t) follows after the entries
    };
    #pragma pack(pop)

    #pragma pack(push, 1)
    union TagValue {
        uint32_t offset; // Offset to data if count*size > 4 bytes
        uint8_t  byte[4];  // For up to 4 BYTE values
        char     ascii_chars[4]; // For up to 4 ASCII chars
        uint16_t short_val[2]; // For up to 2 SHORT values
        uint32_t long_val;    // For 1 LONG value
        int8_t   sbyte[4]; // For up to 4 SBYTE values
        int16_t  sshort[2]; // For up to 2 SSHORT values
        int32_t  slong;   // For 1 SLONG value
        float    float_val; // For 1 FLOAT value
    };

    #pragma pack(push, 1)
    struct TiffTag {
        uint16_t code;       // Tag identifier
        uint16_t datatype;   // Data type (1=BYTE, 2=ASCII, 3=SHORT, etc.)
        uint32_t count;      // Number of values
        TagValue value;      // Value or offset to value
    };
    #pragma pack(pop)
    """
    cdef struct TiffHeader:
        int8_t[2] byteOrder
        uint16_t version
        uint32_t firstIFDOffset

    cdef struct IFDHeader:
        uint16_t numEntries

    cdef union TagValue:
        uint32_t offset
        uint8_t[4] byte
        int8_t[4] ascii_chars
        uint16_t[2] short_val
        uint32_t long_val
        int8_t[4] sbyte
        int16_t[2] sshort
        int32_t slong
        float float_val

    cdef struct TiffTag:
        uint16_t code
        uint16_t datatype
        uint32_t count
        TagValue value


@cython.final
cdef class CustomTiffReader:
    """
    A reader with state that efficiently reads TIFF files
    using targeted file access.

    Not thread safe.
    """

    cdef FILE* file_handle
    cdef int64_t file_size

    # page information
    cdef vector[uint32_t] databytecounts
    cdef vector[uint32_t] dataoffsets
    cdef uint32_t image_width
    cdef uint32_t image_length
    cdef uint32_t tile_width
    cdef uint32_t tile_length
    cdef uint16_t bitspersample

    # ZSTD context
    cdef ZSTD_DCtx* zstd_dctx
    
    cdef uint8_t[::1] input_buffer_view


    def __cinit__(self):
        self.file_handle = NULL
        self.zstd_dctx = ZSTD_createDCtx()
        if self.zstd_dctx == NULL:
            raise MemoryError("Failed to create ZSTD decompression context")

    def __dealloc__(self):
        if self.file_handle != NULL:
            fclose(self.file_handle)
            self.file_handle = NULL
        
        # Free ZSTD context
        if self.zstd_dctx != NULL:
            ZSTD_freeDCtx(self.zstd_dctx)

    cdef bint _open(self, string &path) noexcept nogil:
        """
        Opens the target file using standard C file I/O
        """
        # Close previously opened file
        if self.file_handle != NULL:
            fclose(self.file_handle)
            self.file_handle = NULL
            
        # Open the file
        self.file_handle = fopen(path.c_str(), "rb")
        return self.file_handle != NULL

    cdef bint _check_header(self) noexcept nogil:
        """
        Check the header is classic tiff little-endian
        """
        cdef TiffHeader header

        fseek(self.file_handle, 0, SEEK_SET)
        if fread(&header, sizeof(TiffHeader), 1, self.file_handle) != 1:
            return False

        if header.byteOrder[0] != header.byteOrder[1] and\
           header.byteOrder[0] != 73: # 'I'
            return False
        return header.version == 42

    cdef uint32_t _get_offset_first_page(self) noexcept nogil:
        """
        Retrieve from the header the offset to the first page
        """
        cdef TiffHeader header

        fseek(self.file_handle, 0, SEEK_SET)
        fread(&header, sizeof(TiffHeader), 1, self.file_handle)
        return header.firstIFDOffset

    cdef uint32_t _get_offset_next_page(self, uint32_t cur_page_offset) noexcept nogil:
        """
        Assuming cur_page_offset points to a page, skips the tags
        and retrieve the offset to the next page and returns it
        """
        cdef IFDHeader header

        fseek(self.file_handle, cur_page_offset, SEEK_SET)
        if fread(&header, sizeof(IFDHeader), 1, self.file_handle) != 1:
            return 0

        # Skip the tags
        fseek(self.file_handle, header.numEntries * sizeof(TiffTag), SEEK_CUR)

        # Next IFD offset is stored after all the tag entries
        cdef uint32_t next_ifd_offset
        if fread(&next_ifd_offset, sizeof(uint32_t), 1, self.file_handle) != 1:
            return 0
        return next_ifd_offset  # Return the offset to the next IFD, 0 if this is the last one

    cdef bint _parse_data_for_reading(self, uint32_t page_offset) noexcept nogil:
        """
        Go through the page tags and retrieves the values of the tags that
        matter for our usage
        """
        cdef IFDHeader header

        fseek(self.file_handle, page_offset, SEEK_SET)
        if fread(&header, sizeof(IFDHeader), 1, self.file_handle) != 1:
            return False

        cdef uint16_t num_tags = header.numEntries
        cdef TiffTag* tags = <TiffTag*>malloc(num_tags * sizeof(TiffTag))
        if fread(tags, num_tags * sizeof(TiffTag), 1, self.file_handle) != 1:
            return False

        # We rely on the fact indices are support to only go increasing
        # image_width: 256 (ImageWidth)
        # image_length: 257 (ImageLength)
        # bitspersample: 258
        # compression: 259 (must be 34926 or 50000)
        # tilewidth: 322 (TileWidth)
        # tilelength: 323 (TileHeight)
        # dataoffsets: 324 (TileOffsets)
        # databytecounts: 325 (TileByteCounts)
        # sampleformat: 339 (must be 1 or undefined)
        # ImageDepth: 32997 (must be 1 or undefined)
        # TileDepth: 32998 (must be 1 or undefined)

        while tags[0].code < 256:
            tags += 1
            num_tags -= 1
            if num_tags == 0:
                return False

        if tags[0].datatype != 4 or tags[0].count != 1:
            return False
        self.image_width = tags[0].value.long_val
        tags += 1
        num_tags -= 1

        if num_tags == 0 or tags[0].code != 257:
            return False
        if tags[0].datatype != 4 or tags[0].count != 1:
            return False
        self.image_length = tags[0].value.long_val
        tags += 1
        num_tags -= 1

        if num_tags == 0 or tags[0].code != 258:
            return False
        if tags[0].datatype != 3 or tags[0].count != 1:
            return False
        self.bitspersample = tags[0].value.short_val[0]
        if self.bitspersample != 8 and self.bitspersample != 16:
            return False
        tags += 1
        num_tags -= 1

        if num_tags == 0 or tags[0].code != 259:
            return False
        if tags[0].datatype != 3 or tags[0].count != 1:
            return False
        if tags[0].value.short_val[0] != 34926 and tags[0].value.short_val[0] != 50000:
            return False
        tags += 1
        num_tags -= 1
        if num_tags == 0:
            return False

        while tags[0].code < 322:
            tags += 1
            num_tags -= 1
            if num_tags == 0:
                return False

        if tags[0].code != 322:
            return False
        if tags[0].datatype != 4 or tags[0].count != 1:
            return False
        self.tile_width = tags[0].value.long_val
        tags += 1
        num_tags -= 1

        if num_tags == 0 or tags[0].code != 323:
            return False
        if tags[0].datatype != 4 or tags[0].count != 1:
            return False
        self.tile_length = tags[0].value.long_val
        tags += 1
        num_tags -= 1

        if num_tags == 0 or tags[0].code != 324:
            return False
        if tags[0].datatype != 4:
            return False
        self.dataoffsets.resize(tags[0].count)
        # Case of count == 1
        if tags[0].count == 1:
            self.dataoffsets[0] = tags[0].value.offset
        else:
            fseek(self.file_handle, tags[0].value.offset, SEEK_SET)
            if fread(&self.dataoffsets[0],
                     tags[0].count * sizeof(uint32_t),
                     1,
                     self.file_handle) != 1:
                return False
        tags += 1
        num_tags -= 1

        if num_tags == 0 or tags[0].code != 325:
            return False
        if tags[0].datatype != 4:
            return False
        self.databytecounts.resize(tags[0].count)
        # Case of count == 1
        if tags[0].count == 1:
            self.databytecounts[0] = tags[0].value.offset
        else:
            fseek(self.file_handle, tags[0].value.offset, SEEK_SET)
            if fread(&self.databytecounts[0],
                     tags[0].count * sizeof(uint32_t),
                     1,
                     self.file_handle) != 1:
                return False
        tags += 1
        num_tags -= 1

        if num_tags == 0:
            return True

        while tags[0].code < 339:
            tags += 1
            num_tags -= 1
            if num_tags == 0:
                return True

        if tags[0].code == 339:
            if tags[0].datatype != 3 or tags[0].count != 1:
                return False
            if tags[0].value.short_val[0] != 1:
                return False
            tags += 1
            num_tags -= 1

        if num_tags == 0:
            return True

        while tags[0].code < 32997:
            tags += 1
            num_tags -= 1
            if num_tags == 0:
                return True

        if tags[0].code == 32997:
            if tags[0].datatype != 3 or tags[0].count != 1:
                return False
            if tags[0].value.short_val[0] != 1:
                return False
            tags += 1
            num_tags -= 1

        if num_tags == 0:
            return True

        if tags[0].code == 32998:
            if tags[0].datatype != 3 or tags[0].count != 1:
                return False
            if tags[0].value.short_val[0] != 1:
                return False
            tags += 1
            num_tags -= 1

        return True

    cdef void* _decode_uint16(self,
                              uint32_t start_x,
                              uint32_t start_y,
                              uint32_t crop_width,
                              uint32_t crop_height) noexcept nogil:
        """
        Decode to output_buffer the designated crop
        The crop area has already been checked to be in the image
        """
        # Calculate tile indices for the requested region
        cdef uint32_t start_tile_x = start_x // self.tile_width
        cdef uint32_t start_tile_y = start_y // self.tile_length
        cdef uint32_t end_tile_x = (start_x + crop_width + self.tile_width - 1) // self.tile_width
        cdef uint32_t end_tile_y = (start_y + crop_height + self.tile_length - 1) // self.tile_length

        # Calculate number of tiles across
        cdef uint32_t tiles_across = (self.image_width + self.tile_width - 1) // self.tile_width

        # Variables for decoding and copying
        cdef uint32_t tile_index, tile_offset, tile_bytecount
        cdef uint32_t this_tile_x, this_tile_y
        cdef int32_t src_start_x, src_start_y, src_width, src_height
        cdef int32_t dst_start_x, dst_start_y, dst_idx
        cdef int32_t x, y, src_idx
        
        # For ZSTD decompression
        cdef uint8_t* compressed_data
        cdef uint16_t* decompressed_data
        cdef size_t decompressed_size, result

        cdef uint16_t* output_buffer = <uint16_t*>malloc(crop_width * crop_height * sizeof(uint16_t))
        if output_buffer == NULL:
            return NULL

        # Calculate decompressed size
        decompressed_size = self.tile_width * self.tile_length * sizeof(uint16_t)
                    
        # Allocate buffer for decompressed data
        decompressed_data = <uint16_t*>malloc(decompressed_size)
        if decompressed_data == NULL:
            free(output_buffer)
            return NULL

        # Allocate buffer for compressed data,
        cdef size_t compressed_data_size = 0
        for tile_bytecount in self.databytecounts:
            compressed_data_size = max(tile_bytecount, compressed_data_size)
        compressed_data = <uint8_t*>malloc(compressed_data_size)
        if compressed_data == NULL:
            free(decompressed_data)
            free(output_buffer)
            return NULL

        # Process each relevant tile
        for this_tile_y in range(start_tile_y, end_tile_y):
            for this_tile_x in range(start_tile_x, end_tile_x):
                tile_index = this_tile_y * tiles_across + this_tile_x
                
                # Check if the tile is within the image bounds
                if (tile_index < self.dataoffsets.size() and 
                    this_tile_x < (self.image_width + self.tile_width - 1) // self.tile_width and 
                    this_tile_y < (self.image_length + self.tile_length - 1) // self.tile_length):
                    
                    tile_offset = self.dataoffsets[tile_index]
                    tile_bytecount = self.databytecounts[tile_index]
                    
                    if tile_bytecount == 0:
                        continue
                    
                    # Retrieve compressed data
                    fseek(self.file_handle, tile_offset, SEEK_SET)
                    if fread(compressed_data, tile_bytecount, 1, self.file_handle) != 1:
                        free(decompressed_data)
                        free(output_buffer)
                        return NULL
                            
                    # Decompress data directly using ZSTD
                    result = ZSTD_decompressDCtx(
                        self.zstd_dctx,
                        decompressed_data,
                        decompressed_size,
                        compressed_data,
                        tile_bytecount
                    )
                    
                    if ZSTD_isError(result):
                        free(decompressed_data)
                        free(output_buffer)
                        return NULL
                    
                    # Apply delta decoding directly on decompressed data
                    delta_decode_uint16(decompressed_data, self.tile_width, self.tile_length, self.tile_width)
                    
                    # Calculate source and destination positions
                    # dst_start_x/y represent the relative start of the
                    # tile in the output buffer
                    dst_start_x = this_tile_x * self.tile_width - start_x
                    dst_start_y = this_tile_y * self.tile_length - start_y
                    
                    # src_start_x/y represent the relative start of the
                    # data to copy from the tile
                    src_start_x = max(0, -dst_start_x)
                    src_start_y = max(0, -dst_start_y)

                    # Calculate effective tile dimensions
                    # src_width/height represent the number of pixels to copy
                    # from the tile
                    src_width = min(self.tile_width - src_start_x, crop_width - max(0, dst_start_x))
                    src_height = min(self.tile_length - src_start_y, crop_height - max(0, dst_start_y))
                    
                    if src_width > 0 and src_height > 0:
                        # Copy the decoded region to the output buffer
                        for y in range(src_height):
                            src_idx = (src_start_y + y) * self.tile_width + src_start_x
                            dst_idx = (max(0, dst_start_y) + y) * crop_width + max(0, dst_start_x)
                            memcpy(
                                &output_buffer[dst_idx],
                                &decompressed_data[src_idx],
                                src_width * sizeof(uint16_t)
                            )
                    
        # Cleanup
        free(decompressed_data)
        free(compressed_data)

        return output_buffer

    cdef void* _decode_uint8(self,
                             uint32_t start_x,
                             uint32_t start_y,
                             uint32_t crop_width,
                             uint32_t crop_height) noexcept nogil:
        """
        Decode to output_buffer the designated crop
        The crop area has already been checked to be in the image
        """
        # Calculate tile indices for the requested region
        cdef uint32_t start_tile_x = start_x // self.tile_width
        cdef uint32_t start_tile_y = start_y // self.tile_length
        cdef uint32_t end_tile_x = (start_x + crop_width + self.tile_width - 1) // self.tile_width
        cdef uint32_t end_tile_y = (start_y + crop_height + self.tile_length - 1) // self.tile_length

        # Calculate number of tiles across
        cdef uint32_t tiles_across = (self.image_width + self.tile_width - 1) // self.tile_width

        # Variables for decoding and copying
        cdef uint32_t tile_index, tile_offset, tile_bytecount
        cdef uint32_t this_tile_x, this_tile_y
        cdef int32_t src_start_x, src_start_y, src_width, src_height
        cdef int32_t dst_start_x, dst_start_y, dst_idx
        cdef int32_t x, y, src_idx
        
        # For ZSTD decompression
        cdef uint8_t* compressed_data
        cdef uint8_t* decompressed_data
        cdef size_t decompressed_size, result

        cdef uint8_t* output_buffer = <uint8_t*>malloc(crop_width * crop_height * sizeof(uint8_t))
        if output_buffer == NULL:
            return NULL

        # Calculate decompressed size
        decompressed_size = self.tile_width * self.tile_length * sizeof(uint8_t)

        # Allocate buffer for decompressed data
        decompressed_data = <uint8_t*>malloc(decompressed_size)
        if decompressed_data == NULL:
            free(output_buffer)
            return NULL

        # Allocate buffer for compressed data,
        cdef size_t compressed_data_size = 0
        for tile_bytecount in self.databytecounts:
            compressed_data_size = max(tile_bytecount, compressed_data_size)
        compressed_data = <uint8_t*>malloc(compressed_data_size)
        if compressed_data == NULL:
            free(decompressed_data)
            free(output_buffer)
            return NULL

        # Process each relevant tile
        for this_tile_y in range(start_tile_y, end_tile_y):
            for this_tile_x in range(start_tile_x, end_tile_x):
                tile_index = this_tile_y * tiles_across + this_tile_x
                
                # Check if the tile is within the image bounds
                if (tile_index < self.dataoffsets.size() and 
                    this_tile_x < (self.image_width + self.tile_width - 1) // self.tile_width and 
                    this_tile_y < (self.image_length + self.tile_length - 1) // self.tile_length):
                    
                    tile_offset = self.dataoffsets[tile_index]
                    tile_bytecount = self.databytecounts[tile_index]
                    
                    if tile_bytecount == 0:
                        continue
                    
                    # Retrieve compressed data
                    fseek(self.file_handle, tile_offset, SEEK_SET)
                    if fread(compressed_data, tile_bytecount, 1, self.file_handle) != 1:
                        free(decompressed_data)
                        free(output_buffer)
                        return NULL
    
                    # Decompress data directly using ZSTD
                    result = ZSTD_decompressDCtx(
                        self.zstd_dctx,
                        decompressed_data,
                        decompressed_size,
                        compressed_data,
                        tile_bytecount
                    )
                    
                    if ZSTD_isError(result):
                        free(decompressed_data)
                        free(output_buffer)
                        return NULL
                    
                    # Apply delta decoding directly on decompressed data
                    delta_decode_uint8(decompressed_data, self.tile_width, self.tile_length, self.tile_width)
                    
                    # Calculate source and destination positions
                    dst_start_x = this_tile_x * self.tile_width - start_x
                    dst_start_y = this_tile_y * self.tile_length - start_y
                    
                    # Calculate effective tile dimensions
                    src_start_x = max(0, -dst_start_x)
                    src_start_y = max(0, -dst_start_y)
                    src_width = min(self.tile_width - src_start_x, crop_width - max(0, dst_start_x))
                    src_height = min(self.tile_length - src_start_y, crop_height - max(0, dst_start_y))
                    
                    if src_width > 0 and src_height > 0:
                        # Copy the decoded region to the output buffer
                        for y in range(src_height):
                            src_idx = (src_start_y + y) * self.tile_width + src_start_x
                            dst_idx = (max(0, dst_start_y) + y) * crop_width + max(0, dst_start_x)
                            memcpy(
                                &output_buffer[dst_idx],
                                &decompressed_data[src_idx],
                                src_width * sizeof(uint8_t)
                            )
                    
        # Cleanup
        free(decompressed_data)
        free(compressed_data)

        return <void*>output_buffer

    cpdef object read(self, str path, int start_y, int start_x, int stop_y, int stop_x, int page):
        """
        Reads the designated crop from the designated page,
        returns a numpy array of the requested shape, or raises
        an exception in case of failure
        """
        cdef:
            string path_str
            uint32_t crop_width
            uint32_t crop_height
            uint32_t first_page_offset
            uint32_t cur_page_offset
            uint32_t cur_page

        path_str = path.encode('utf-8')
        cdef void *dst_ptr
        with nogil:
            if not self._open(path_str):
                raise ValueError("Failed to open file")

            fseek(self.file_handle, 0, SEEK_END)
            self.file_size = ftell(self.file_handle)
            if self.file_size == -1:
                raise ValueError("Failed to get file size")
            if not self._check_header():
                raise ValueError("Unsupported TIFF header")

            first_page_offset = self._get_offset_first_page()
            cur_page_offset = first_page_offset
            cur_page = 0
            while cur_page < page:
                cur_page_offset = self._get_offset_next_page(cur_page_offset)
                if cur_page_offset == 0:
                    raise ValueError(f"Requested page not found, only {cur_page} pages found")
                cur_page += 1

            if not self._parse_data_for_reading(cur_page_offset):
                raise ValueError("Failed to parse page tags")

            if stop_x == 0:
                stop_x = self.image_width
            if stop_y == 0:
                stop_y = self.image_length

            if not (stop_x > start_x and stop_x <= self.image_width and
                    stop_y > start_y and stop_y <= self.image_length):
                raise ValueError(f"Invalid crop area, requested ({start_x}, {start_y}) to ({stop_x}, {stop_y}), image is ({self.image_width}, {self.image_length})")
            crop_width = stop_x - start_x
            crop_height = stop_y - start_y

            if self.bitspersample == 16:
                dst_ptr = self._decode_uint16(start_x, start_y, crop_width, crop_height)
                if dst_ptr == NULL:
                    raise ValueError("Failed to decode data")
            else:
                dst_ptr = self._decode_uint8(start_x, start_y, crop_width, crop_height)
                if dst_ptr == NULL:
                    raise ValueError("Failed to decode data")
        cdef array output_array
        if self.bitspersample == 16:
            output_array = array((crop_height, crop_width), itemsize=2, allocate_buffer=False, format='H')
            output_array.data = <char*>dst_ptr
            output_array.callback_free_data = free
        else:
            output_array = array((crop_height, crop_width), itemsize=1, allocate_buffer=False, format='B')
            output_array.data = <char*>dst_ptr
            output_array.callback_free_data = free
        return output_array

    cpdef tuple get_image_dimensions(self, str path, int page):
        """
        Returns the image dimensions of the designated page
        (height, width)
        """
        cdef:
            string path_str
            uint32_t first_page_offset
            uint32_t cur_page_offset
            uint32_t cur_page

        path_str = path.encode('utf-8')
        
        with nogil:
            if not self._open(path_str):
                with gil:
                    raise ValueError("Failed to open file")

            fseek(self.file_handle, 0, SEEK_END)
            self.file_size = ftell(self.file_handle)
            if self.file_size == -1:
                with gil:
                    raise ValueError("Failed to get file size")
            if not self._check_header():
                with gil:
                    raise ValueError("Unsupported TIFF header")

            first_page_offset = self._get_offset_first_page()
            cur_page_offset = first_page_offset
            cur_page = 0
            while cur_page < page:
                cur_page_offset = self._get_offset_next_page(cur_page_offset)
                if cur_page_offset == 0:
                    with gil:
                        raise ValueError(f"Requested page not found, only {cur_page} pages found")
                cur_page += 1

            if not self._parse_data_for_reading(cur_page_offset):
                with gil:
                    raise ValueError("Failed to parse page tags")
        
        # Return dimensions as (height, width)
        return (self.image_length, self.image_width)





