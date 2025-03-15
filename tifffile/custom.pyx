#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

import threading
import zstandard as zstd
import numpy as np

from libc.stdint cimport int64_t, uint16_t, uint8_t, uint64_t
from cython.view cimport array
from libc.string cimport memcpy
from libcpp.vector cimport vector

from .files cimport FileHandle
from .pages cimport TiffPage
from .tags cimport TiffTags

LOCAL_STATE = threading.local()

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

cdef class FastDecoderZstdDelta16:
    """
    Fast decoding of tiff page with zstd compression and delta encoding.
    Supports only 3D uint16 data.
    """
    cdef object zstd_decompressor
    cdef uint8_t[::1] input_buffer
    cdef uint64_t[::1] segment_info
    cdef uint64_t[::1] decompressed_sizes

    def __cinit__(self):
        self.zstd_decompressor = zstd.ZstdDecompressor()
        self.input_buffer = array(shape=(4*1024*1024,), itemsize=1, format="B")
        self.segment_info = array(shape=(16*1024,), itemsize=8, format="Q")
        self.decompressed_sizes = array(shape=(8*1024,), itemsize=8, format="Q")


    cdef run(self, TiffPage page, int64_t start_x, int64_t stop_x, int64_t start_y, int64_t stop_y):
        """
        Run fast decoding of tiff page with zstd compression and delta encoding.
        """
        cdef int64_t width = stop_x - start_x
        cdef int64_t height = stop_y - start_y
            
        # Allocate output array
        cdef object output_array = np.empty((height, width), dtype=np.uint16)
        cdef uint16_t[::1] output_buffer = output_array.reshape(-1)

        cdef FileHandle filehandle = page.fh
        
        # Calculate tile indices for the requested region
        cdef int64_t tile_width = page.tilewidth
        cdef int64_t tile_length = page.tilelength
        cdef int64_t start_tile_x = start_x // tile_width
        cdef int64_t start_tile_y = start_y // tile_length
        cdef int64_t end_tile_x = (stop_x + tile_width - 1) // tile_width
        cdef int64_t end_tile_y = (stop_y + tile_length - 1) // tile_length
        
        # Indices within the tiles
        cdef int64_t rel_start_x = start_x % tile_width
        cdef int64_t rel_start_y = start_y % tile_length

        cdef int64_t tiles_across = (page.imagewidth + tile_width - 1) // tile_width

        # Variables for decoding and copying
        cdef int64_t tile_index, tile_offset, tile_bytecount
        cdef int64_t this_tile_x, this_tile_y
        cdef int64_t src_start_x, src_start_y, src_width, src_height
        cdef int64_t dst_start_x, dst_start_y
        cdef int64_t x, y, src_idx, dst_idx, row_size, max_bytecount = 0
        
        # C++ vectors to store tile information
        cdef vector[int64_t] tile_x_vec
        cdef vector[int64_t] tile_y_vec

        # Allocate the buffers
        # input buffer and temporary buffer: we reuse previous data
        # if it exists, and we overallocate a bit to avoid reallocations
        if self.input_buffer.shape[0] < 8 * width * height:
            self.input_buffer = array(shape=(8 * width * height,), itemsize=1, format="B")
        # Pre-calculate the number of tiles and pre-allocate vectors
        cdef int64_t tile_count = (end_tile_x - start_tile_x) * (end_tile_y - start_tile_y)
        if self.segment_info.shape[0] < 2 * tile_count:
            # uint64_t * 2 storage for BufferWithSegments
            self.segment_info = array(shape=(2 * tile_count,), itemsize=8, format="Q")
            self.decompressed_sizes = array(shape=(tile_count,), itemsize=8, format="Q")

        cdef int n_segment = 0
        cdef int64_t segment_offset = 0
        with nogil:
            tile_x_vec.reserve(tile_count)
            tile_y_vec.reserve(tile_count)
            for this_tile_y in range(start_tile_y, end_tile_y):
                for this_tile_x in range(start_tile_x, end_tile_x):
                    tile_index = this_tile_y * tiles_across + this_tile_x

                    # Check if the tile is within the image bounds
                    if (tile_index < page._dataoffsets.size() and 
                        this_tile_x < page.imagewidth and 
                        this_tile_y < page.imagelength):
                        
                        tile_offset = page._dataoffsets[tile_index]
                        tile_bytecount = page._databytecounts[tile_index]
                        
                        if tile_bytecount == 0:
                            continue

                        # Very unlikely but lets be safe
                        if segment_offset + tile_bytecount > self.input_buffer.shape[0]:
                            with gil:
                                raise ValueError("Input buffer too small")

                        # Retrieve the tile data
                        filehandle.read_into(&self.input_buffer[segment_offset],
                                             tile_offset,
                                             tile_bytecount)

                        # Store the segment information
                        tile_x_vec.push_back(this_tile_x)
                        tile_y_vec.push_back(this_tile_y)
                        self.segment_info[2*n_segment] = segment_offset
                        self.segment_info[2*n_segment + 1] = tile_bytecount
                        self.decompressed_sizes[n_segment] = tile_width * tile_length * sizeof(uint16_t)
                        segment_offset += tile_bytecount
                        n_segment += 1

        # Convert to BufferWithSegments
        buffer_with_segments = zstd.BufferWithSegments(self.input_buffer, self.segment_info[:2*n_segment])

        # Decompress the data
        decompressed_data = self.zstd_decompressor.multi_decompress_to_buffer(
            buffer_with_segments,
            self.decompressed_sizes[:n_segment]
        )

        # decompressed_data is a BufferWithSegmentsCollection
        cdef vector[uint8_t*] decompressed_buffer_pointers
        cdef const uint8_t[::1] decompressed_buffer
        cdef int i
        for buffer in decompressed_data:
            decompressed_buffer = buffer
            decompressed_buffer_pointers.push_back(<uint8_t*><void*>&decompressed_buffer[0])

        assert decompressed_buffer_pointers.size() == n_segment

        cdef int64_t max_tile_size = tile_width * tile_length
        cdef uint16_t* src_data 
        with nogil:
            for i in range(n_segment):
                # Get tile coordinates
                this_tile_x = tile_x_vec[i]
                this_tile_y = tile_y_vec[i]
                
                # Calculate source and destination positions
                dst_start_x = this_tile_x * tile_width - start_x
                dst_start_y = this_tile_y * tile_length - start_y
                
                # Calculate effective tile dimensions
                src_start_x = max(0, -dst_start_x)
                src_start_y = max(0, -dst_start_y)
                src_width = min(tile_width - src_start_x, width - dst_start_x - src_start_x)
                src_height = min(tile_length - src_start_y, height - dst_start_y - src_start_y)
                
                if src_width <= 0 or src_height <= 0:
                    continue
                
                # Get pointer to decompressed data
                src_data = <uint16_t*>decompressed_buffer_pointers[i]
                
                # Apply delta decoding directly on source data
                delta_decode_uint16(src_data, tile_width, tile_length, tile_width)
                
                # Copy the decoded region to the output buffer
                for y in range(src_height):
                    dst_idx = (dst_start_y + y + src_start_y) * width + (dst_start_x + src_start_x)
                    src_idx = (src_start_y + y) * tile_width + src_start_x
                    memcpy(&output_buffer[dst_idx], &src_data[src_idx], src_width * sizeof(uint16_t))

        return output_array
    

cdef class FastDecoderZstdDelta8:
    """
    Fast decoding of tiff page with zstd compression and delta encoding.
    Supports only 3D uint8 data.
    """
    cdef object zstd_decompressor
    cdef uint8_t[::1] input_buffer
    cdef uint64_t[::1] segment_info
    cdef uint64_t[::1] decompressed_sizes

    def __cinit__(self):
        self.zstd_decompressor = zstd.ZstdDecompressor()
        self.input_buffer = array(shape=(4*1024*1024,), itemsize=1, format="B")
        self.segment_info = array(shape=(16*1024,), itemsize=8, format="Q")
        self.decompressed_sizes = array(shape=(8*1024,), itemsize=8, format="Q")


    cdef run(self, TiffPage page, int64_t start_x, int64_t stop_x, int64_t start_y, int64_t stop_y):
        """
        Run fast decoding of tiff page with zstd compression and delta encoding.
        """
        cdef int64_t width = stop_x - start_x
        cdef int64_t height = stop_y - start_y
            
        # Allocate output array
        cdef object output_array = np.empty((height, width), dtype=np.uint8)
        cdef uint8_t[::1] output_buffer = output_array.reshape(-1)

        cdef FileHandle filehandle = page.fh
        
        # Calculate tile indices for the requested region
        cdef int64_t tile_width = page.tilewidth
        cdef int64_t tile_length = page.tilelength
        cdef int64_t start_tile_x = start_x // tile_width
        cdef int64_t start_tile_y = start_y // tile_length
        cdef int64_t end_tile_x = (stop_x + tile_width - 1) // tile_width
        cdef int64_t end_tile_y = (stop_y + tile_length - 1) // tile_length
        
        # Indices within the tiles
        cdef int64_t rel_start_x = start_x % tile_width
        cdef int64_t rel_start_y = start_y % tile_length

        cdef int64_t tiles_across = (page.imagewidth + tile_width - 1) // tile_width

        # Variables for decoding and copying
        cdef int64_t tile_index, tile_offset, tile_bytecount
        cdef int64_t this_tile_x, this_tile_y
        cdef int64_t src_start_x, src_start_y, src_width, src_height
        cdef int64_t dst_start_x, dst_start_y
        cdef int64_t x, y, src_idx, dst_idx, row_size, max_bytecount = 0
        
        # C++ vectors to store tile information
        cdef vector[int64_t] tile_x_vec
        cdef vector[int64_t] tile_y_vec

        # Allocate the buffers
        # input buffer and temporary buffer: we reuse previous data
        # if it exists, and we overallocate a bit to avoid reallocations
        if self.input_buffer.shape[0] < 4 * width * height:  # 4x instead of 8x since each pixel is 1 byte
            self.input_buffer = array(shape=(4 * width * height,), itemsize=1, format="B")
        # Pre-calculate the number of tiles and pre-allocate vectors
        cdef int64_t tile_count = (end_tile_x - start_tile_x) * (end_tile_y - start_tile_y)
        if self.segment_info.shape[0] < 2 * tile_count:
            # uint64_t * 2 storage for BufferWithSegments
            self.segment_info = array(shape=(2 * tile_count,), itemsize=8, format="Q")
            self.decompressed_sizes = array(shape=(tile_count,), itemsize=8, format="Q")

        cdef int n_segment = 0
        cdef int64_t segment_offset = 0
        with nogil:
            tile_x_vec.reserve(tile_count)
            tile_y_vec.reserve(tile_count)
            for this_tile_y in range(start_tile_y, end_tile_y):
                for this_tile_x in range(start_tile_x, end_tile_x):
                    tile_index = this_tile_y * tiles_across + this_tile_x

                    # Check if the tile is within the image bounds
                    if (tile_index < page._dataoffsets.size() and 
                        this_tile_x < page.imagewidth and 
                        this_tile_y < page.imagelength):
                        
                        tile_offset = page._dataoffsets[tile_index]
                        tile_bytecount = page._databytecounts[tile_index]
                        
                        if tile_bytecount == 0:
                            continue

                        # Very unlikely but lets be safe
                        if segment_offset + tile_bytecount > self.input_buffer.shape[0]:
                            with gil:
                                raise ValueError("Input buffer too small")

                        # Retrieve the tile data
                        filehandle.read_into(&self.input_buffer[segment_offset],
                                             tile_offset,
                                             tile_bytecount)

                        # Store the segment information
                        tile_x_vec.push_back(this_tile_x)
                        tile_y_vec.push_back(this_tile_y)
                        self.segment_info[2*n_segment] = segment_offset
                        self.segment_info[2*n_segment + 1] = tile_bytecount
                        self.decompressed_sizes[n_segment] = tile_width * tile_length * sizeof(uint8_t)
                        segment_offset += tile_bytecount
                        n_segment += 1

        # Convert to BufferWithSegments
        buffer_with_segments = zstd.BufferWithSegments(self.input_buffer, self.segment_info[:2*n_segment])

        # Decompress the data
        decompressed_data = self.zstd_decompressor.multi_decompress_to_buffer(
            buffer_with_segments,
            self.decompressed_sizes[:n_segment]
        )

        # decompressed_data is a BufferWithSegmentsCollection
        cdef vector[uint8_t*] decompressed_buffer_pointers
        cdef const uint8_t[::1] decompressed_buffer
        cdef int i
        for buffer in decompressed_data:
            decompressed_buffer = buffer
            decompressed_buffer_pointers.push_back(<uint8_t*><void*>&decompressed_buffer[0])

        assert decompressed_buffer_pointers.size() == n_segment

        cdef int64_t max_tile_size = tile_width * tile_length
        cdef uint8_t* src_data 
        with nogil:
            for i in range(n_segment):
                # Get tile coordinates
                this_tile_x = tile_x_vec[i]
                this_tile_y = tile_y_vec[i]
                
                # Calculate source and destination positions
                dst_start_x = this_tile_x * tile_width - start_x
                dst_start_y = this_tile_y * tile_length - start_y
                
                # Calculate effective tile dimensions
                src_start_x = max(0, -dst_start_x)
                src_start_y = max(0, -dst_start_y)
                src_width = min(tile_width - src_start_x, width - dst_start_x - src_start_x)
                src_height = min(tile_length - src_start_y, height - dst_start_y - src_start_y)
                
                if src_width <= 0 or src_height <= 0:
                    continue
                
                # Get pointer to decompressed data
                src_data = <uint8_t*>decompressed_buffer_pointers[i]
                
                # Apply delta decoding directly on source data
                delta_decode_uint8(src_data, tile_width, tile_length, tile_width)
                
                # Copy the decoded region to the output buffer
                for y in range(src_height):
                    dst_idx = (dst_start_y + y + src_start_y) * width + (dst_start_x + src_start_x)
                    src_idx = (src_start_y + y) * tile_width + src_start_x
                    memcpy(&output_buffer[dst_idx], &src_data[src_idx], src_width * sizeof(uint8_t))

        return output_array


cpdef object fast_parsing_zstd_delta(
    TiffPage page,
    int64_t start_x,
    int64_t stop_x,
    int64_t start_y,
    int64_t stop_y):
    """
    Fast decoding of tiff page with zstd compression and delta encoding.
    
    Parameters:
        page: TiffPage to decode
        start_x: Start X coordinate (inclusive)
        stop_x: Stop X coordinate (exclusive)
        start_y: Start Y coordinate (inclusive)
        stop_y: Stop Y coordinate (exclusive)
        
    Returns:
        Decoded 2D array with shape (stop_y - start_y, stop_x - start_x)
    """
    # Check if the page is compatible with our decoder
    if not page.is_tiled():
        raise ValueError("Page must be tiled")
    if page.compression != 50000 and page.compression != 34926:
        raise ValueError(f"Page compression must be ZSTD (50000 or 34926), got {page.compression}")
    if page.bitspersample != 16 and page.bitspersample != 8:
        raise ValueError(f"BitSampleFormat must be 8 or 16, got {page.bitspersample}")
    if page.sampleformat != 1:  # 1 is UINT
        raise ValueError(f"SampleFormat must be 1 (UINT), got {page.sampleformat}")
    if page.predictor != 2:  # 2 is HORIZONTAL differencing
        raise ValueError(f"Predictor must be 2 (HORIZONTAL), got {page.predictor}")
    
    # Get or create the decoder instance
    if not hasattr(LOCAL_STATE, 'zstd_delta_decoder_uint16'):
        LOCAL_STATE.zstd_delta_decoder_uint16 = FastDecoderZstdDelta16()

    # Get or create the decoder instance
    if not hasattr(LOCAL_STATE, 'zstd_delta_decoder_uint8'):
        LOCAL_STATE.zstd_delta_decoder_uint8 = FastDecoderZstdDelta8()
    
    # Run the decoder
    cdef FastDecoderZstdDelta8 decoder_uint8
    cdef FastDecoderZstdDelta16 decoder_uint16

    if page.bitspersample == 16:
        decoder_uint16 = LOCAL_STATE.zstd_delta_decoder_uint16
        return decoder_uint16.run(page, start_x, stop_x, start_y, stop_y)
    else:
        decoder_uint8 = LOCAL_STATE.zstd_delta_decoder_uint8
        return decoder_uint8.run(page, start_x, stop_x, start_y, stop_y)

