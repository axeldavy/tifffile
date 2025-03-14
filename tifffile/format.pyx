#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

from libc.stdint cimport uint8_t, int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
import struct
from cpython.bytes cimport PyBytes_AS_STRING

from .utils import indent
from .types cimport DATATYPE

cdef class TiffFormat:
    """TIFF format properties."""

    cpdef bint is_bigtiff(self):
        """Format is 64-bit BigTIFF."""
        return self.version == 43

    cpdef bint is_ndpi(self):
        """Format is 32-bit TIFF with 64-bit offsets used by NDPI."""
        return self.version == 42 and self.offsetsize == 8
        
    @property
    def byteorder_str(self) -> str:
        """Get byteorder as string ('<' or '>')."""
        return '<' if self.byteorder == ByteOrder.II else '>'

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        bits = '32' if self.version == 42 else '64'
        endian = 'little' if self.byteorder == ByteOrder.II else 'big'
        ndpi = ' with 64-bit offsets' if self.is_ndpi() else ''
        return f'<tifffile.TiffFormat {bits}-bit {endian}-endian{ndpi}>'

    def __str__(self) -> str:
        return indent(
            repr(self),
            *(
                f'{self.version}, {self.byteorder_str}, {self.offsetsize}, {self.offsetformat},'
                f'{self.tagnosize}, {self.tagnoformat}, {self.tagsize}, {self.tagformat1},'
                f'{self.tagformat2}, {self.tagoffsetthreshold}'
            ),
        )
        
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil:
        """Parse TIFF tag headers and populate vector of TagHeader structs.
        
        Parameters
        ----------
        v : vector[TagHeader]
            Vector to populate with tag header information.
        headers : bytes
            TIFF tag headers bytes, with multiple tag headers concatenated.
        """
        return # Should be implemented by subclasses

    @staticmethod
    def detect_format(bytes header) -> TiffFormat:
        """Detect appropriate TIFF format from file header.
        
        Parameters
        ----------
        header : bytes
            TIFF file header, must be at least 8 bytes.
            
        Returns
        -------
        TiffFormat
            Appropriate TiffFormat subclass for the detected format.
            
        Raises
        ------
        ValueError
            If the header is not a valid TIFF header.
        """
        cdef:
            const uint8_t* data = <const uint8_t*>PyBytes_AS_STRING(header)
            ByteOrder byteorder
            int64_t version
            uint64_t ifd_offset
        
        # Minimum header size required: 8 bytes
        if len(header) < 8:
            raise ValueError("TIFF header too short")
            
        # Check byte order marker
        if data[0] == 73 and data[1] == 73:  # 'II' - Little-endian
            byteorder = ByteOrder.II
        elif data[0] == 77 and data[1] == 77:  # 'MM' - Big-endian
            byteorder = ByteOrder.MM
        else:
            raise ValueError("Invalid TIFF byte order marker")
        
        # Version depends on byte order
        if byteorder == ByteOrder.II:
            version = data[2] | (data[3] << 8)
        else:
            version = (data[2] << 8) | data[3]
            
        # Check if it's a valid TIFF or BigTIFF version
        if version == 42:  # Classic TIFF
            # Check for NDPI (need more header bytes)
            if len(header) >= 16:
                # In NDPI files, after the IFD count there are 64-bit offsets
                # Check offset size by looking at the IFD offset (should be large)
                if byteorder == ByteOrder.II:
                    ifd_offset = (
                        <uint64_t>data[4] | 
                        (<uint64_t>data[5] << 8) | 
                        (<uint64_t>data[6] << 16) | 
                        (<uint64_t>data[7] << 24) | 
                        (<uint64_t>data[8] << 32) | 
                        (<uint64_t>data[9] << 40) | 
                        (<uint64_t>data[10] << 48) | 
                        (<uint64_t>data[11] << 56)
                    )
                    # Check for the NDPI signature: high offset bytes are non-zero
                    # and offset is reasonable (not extremely large)
                    if (data[8] | data[9] | data[10] | data[11]) and ifd_offset < 0x20000000:
                        return TiffFormatNDPI_LE()
                        
            # Regular 32-bit TIFF
            if byteorder == ByteOrder.II:
                return TiffFormatClassicLE()
            else:
                return TiffFormatClassicBE()
                
        elif version == 43:  # BigTIFF
            # Verify BigTIFF structure (bytesize should be 8, always stored as 2 bytes)
            if len(header) >= 10:
                bytesize = data[4] | (data[5] << 8) if byteorder == ByteOrder.II else (data[4] << 8) | data[5]
                reserved = data[6] | (data[7] << 8) if byteorder == ByteOrder.II else (data[6] << 8) | data[7]
                
                if bytesize != 8 or reserved != 0:
                    raise ValueError("Invalid BigTIFF header")
                    
            if byteorder == ByteOrder.II:
                return TiffFormatBigLE()
            else:
                return TiffFormatBigBE()
        else:
            raise ValueError(f"Unsupported TIFF version: {version}")

cdef class TiffFormatClassicLE(TiffFormat):
    """32-bit little-endian TIFF format."""
    def __cinit__(self) -> None:
        self.version = 42
        self.byteorder = ByteOrder.II
        self.offsetsize = 4
        self.headersize = 8  # 2 bytes byteorder + 2 bytes version + 4 bytes IFD offset
        self.offsetformat = '<I'
        self.tagnosize = 2
        self.tagnoformat = '<H'
        self.tagsize = 12
        self.tagformat1 = '<HH'
        self.tagformat2 = '<I4s'
        self.tagoffsetthreshold = 4
        self._hash = hash((self.version, int(self.byteorder), self.offsetsize))
        
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil:
        """Parse TIFF tag headers and populate vector of TagHeader structs."""
        cdef:
            TagHeader th
            int64_t i, j, num_tags
            uint8_t* ptr
            uint8_t val8
            uint16_t val16
            int16_t sval16
            uint32_t val32
            int32_t sval32
            float* float_ptr
            
        # Calculate number of tags
        num_tags = data_len // self.tagsize
        
        # Reserve space for efficiency
        v.reserve(v.size() + num_tags)
        
        # Process each tag
        for i in range(num_tags):
            # Get pointer to current tag header
            ptr = <uint8_t*>data + (i * self.tagsize)
            
            # Parse fields directly from bytes
            th.code = ptr[0] | (ptr[1] << 8)
            th.datatype = ptr[2] | (ptr[3] << 8)
            th.count = ptr[4] | (ptr[5] << 8) | (ptr[6] << 16) | (ptr[7] << 24)
            
            # Handle the raw value field
            th.value = ptr[8] | (ptr[9] << 8) | (ptr[10] << 16) | (ptr[11] << 24)
            
            # Set interpreted offset value
            th.as_offset = th.value  # For LE, the value is already the offset for 32-bit
            
            # Initialize as_values array
            for j in range(8):
                th.as_values[j].i = 0
            
            # Convert values based on datatype
            if th.count > 0:
                if th.datatype == DATATYPE.BYTE or th.datatype == DATATYPE.ASCII or th.datatype == DATATYPE.UNDEFINED:
                    # 8-bit unsigned types
                    if th.count <= 4:
                        for j in range(min(th.count, 8)):
                            th.as_values[j].i = ptr[8+j]
                
                elif th.datatype == DATATYPE.SBYTE:
                    # 8-bit signed type
                    if th.count <= 4:
                        for j in range(min(th.count, 8)):
                            # Sign extend for negative values
                            val8 = ptr[8+j]
                            if val8 & 0x80:
                                th.as_values[j].i = -1 * ((~val8 & 0x7F) + 1)
                            else:
                                th.as_values[j].i = val8
                
                elif th.datatype == DATATYPE.SHORT:
                    # 16-bit unsigned type
                    if th.count <= 2:
                        for j in range(min(th.count, 8)):
                            if (j*2+1) < 4:  # Ensure we have enough bytes
                                th.as_values[j].i = ptr[8+j*2] | (ptr[8+j*2+1] << 8)
                
                elif th.datatype == DATATYPE.SSHORT:
                    # 16-bit signed type
                    if th.count <= 2:
                        for j in range(min(th.count, 8)):
                            if (j*2+1) < 4:
                                val16 = ptr[8+j*2] | (ptr[8+j*2+1] << 8)
                                # Sign extend
                                if val16 & 0x8000:
                                    th.as_values[j].i = -1 * ((~val16 & 0x7FFF) + 1)
                                else:
                                    th.as_values[j].i = val16
                
                elif th.datatype == DATATYPE.LONG or th.datatype == DATATYPE.IFD:
                    # 32-bit unsigned type
                    if th.count == 1:
                        th.as_values[0].i = th.value
                
                elif th.datatype == DATATYPE.SLONG:
                    # 32-bit signed type
                    if th.count == 1:
                        sval32 = <int32_t>th.value
                        th.as_values[0].i = sval32  # Will automatically sign extend
                
                elif th.datatype == DATATYPE.FLOAT:
                    # 32-bit float - convert from IEEE 754 binary representation
                    if th.count == 1:
                        # Use a float pointer to reinterpret the bytes as float
                        float_ptr = <float*>(&ptr[8])
                        th.as_values[0].d = float_ptr[0]
                
                elif th.datatype == DATATYPE.DOUBLE:
                    # 64-bit double - not stored inline for classic TIFF (requires 8 bytes)
                    pass  # Will need to read from offset
                
                # Other types would require data from the offset
            
            # Add to vector
            v.push_back(th)

cdef class TiffFormatClassicBE(TiffFormat):
    """32-bit big-endian TIFF format."""
    def __cinit__(self) -> None:
        self.version = 42
        self.byteorder = ByteOrder.MM
        self.offsetsize = 4
        self.headersize = 8  # 2 bytes byteorder + 2 bytes version + 4 bytes IFD offset
        self.offsetformat = '>I'
        self.tagnosize = 2
        self.tagnoformat = '>H'
        self.tagsize = 12
        self.tagformat1 = '>HH'
        self.tagformat2 = '>I4s'
        self.tagoffsetthreshold = 4
        self._hash = hash((self.version, int(self.byteorder), self.offsetsize))
        
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil:
        """Parse TIFF tag headers and populate vector of TagHeader structs."""
        cdef:
            TagHeader th
            int64_t i, j, num_tags
            uint8_t* ptr
            uint8_t val8
            uint16_t val16
            int16_t sval16
            uint32_t val32
            int32_t sval32
            float* float_ptr
            uint32_t be_val
            
        # Calculate number of tags
        num_tags = data_len // self.tagsize
        
        # Reserve space for efficiency
        v.reserve(v.size() + num_tags)
        
        # Process each tag
        for i in range(num_tags):
            # Get pointer to current tag header
            ptr = <uint8_t*>data + (i * self.tagsize)
            
            # Parse fields directly from bytes
            th.code = (ptr[0] << 8) | ptr[1]
            th.datatype = (ptr[2] << 8) | ptr[3]
            th.count = (ptr[4] << 24) | (ptr[5] << 16) | (ptr[6] << 8) | ptr[7]
            
            # Handle the raw value field
            th.value = (ptr[8] << 24) | (ptr[9] << 16) | (ptr[10] << 8) | ptr[11]
            
            # Set interpreted offset value
            th.as_offset = th.value  # For BE, the value is already the offset for 32-bit
            
            # Initialize as_values array
            for j in range(8):
                th.as_values[j].i = 0
            
            # Convert values based on datatype
            if th.count > 0:
                if th.datatype == DATATYPE.BYTE or th.datatype == DATATYPE.ASCII or th.datatype == DATATYPE.UNDEFINED:
                    # 8-bit unsigned types
                    if th.count <= 4:
                        for j in range(min(th.count, 8)):
                            th.as_values[j].i = ptr[8+j]
                
                elif th.datatype == DATATYPE.SBYTE:
                    # 8-bit signed type
                    if th.count <= 4:
                        for j in range(min(th.count, 8)):
                            # Sign extend for negative values
                            val8 = ptr[8+j]
                            if val8 & 0x80:
                                th.as_values[j].i = -1 * ((~val8 & 0x7F) + 1)
                            else:
                                th.as_values[j].i = val8
                
                elif th.datatype == DATATYPE.SHORT:
                    # 16-bit unsigned type
                    if th.count <= 2:
                        for j in range(min(th.count, 8)):
                            if (j*2+1) < 4:  # Ensure we have enough bytes
                                th.as_values[j].i = (ptr[8+j*2] << 8) | ptr[8+j*2+1]
                
                elif th.datatype == DATATYPE.SSHORT:
                    # 16-bit signed type
                    if th.count <= 2:
                        for j in range(min(th.count, 8)):
                            if (j*2+1) < 4:
                                val16 = (ptr[8+j*2] << 8) | ptr[8+j*2+1]
                                # Sign extend
                                if val16 & 0x8000:
                                    th.as_values[j].i = -1 * ((~val16 & 0x7FFF) + 1)
                                else:
                                    th.as_values[j].i = val16
                
                elif th.datatype == DATATYPE.LONG or th.datatype == DATATYPE.IFD:
                    # 32-bit unsigned type
                    if th.count == 1:
                        th.as_values[0].i = th.value
                
                elif th.datatype == DATATYPE.SLONG:
                    # 32-bit signed type
                    if th.count == 1:
                        sval32 = <int32_t>th.value
                        th.as_values[0].i = sval32  # Will automatically sign extend
                
                elif th.datatype == DATATYPE.FLOAT:
                    # 32-bit float - convert from IEEE 754 binary representation
                    if th.count == 1:
                        # For big-endian, we need to swap the bytes before interpreting
                        be_val = (ptr[8] << 24) | (ptr[9] << 16) | (ptr[10] << 8) | ptr[11]
                        float_ptr = <float*>(&be_val)
                        th.as_values[0].d = float_ptr[0]
                
                elif th.datatype == DATATYPE.DOUBLE:
                    # 64-bit double - not stored inline for classic TIFF (requires 8 bytes)
                    pass  # Will need to read from offset
                
                # Other types would require data from the offset
            
            # Add to vector
            v.push_back(th)

cdef class TiffFormatBigLE(TiffFormat):
    """64-bit little-endian TIFF format."""
    def __cinit__(self) -> None:
        self.version = 43
        self.byteorder = ByteOrder.II
        self.offsetsize = 8
        self.headersize = 16  # 2 bytes byteorder + 2 bytes version + 2 bytes bytesize + 2 bytes reserved + 8 bytes IFD offset
        self.offsetformat = '<Q'
        self.tagnosize = 8
        self.tagnoformat = '<Q'
        self.tagsize = 20
        self.tagformat1 = '<HH'
        self.tagformat2 = '<Q8s'
        self.tagoffsetthreshold = 8
        self._hash = hash((self.version, int(self.byteorder), self.offsetsize))
        
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil:
        """Parse TIFF tag headers and populate vector of TagHeader structs."""
        cdef:
            TagHeader th
            int64_t i, j, num_tags
            uint8_t* ptr
            uint8_t val8
            uint16_t val16
            int16_t sval16
            uint32_t val32
            int32_t sval32
            uint64_t val64
            int64_t sval64
            float* float_ptr
            double* double_ptr
            
        # Calculate number of tags
        num_tags = data_len // self.tagsize
        
        # Reserve space for efficiency
        v.reserve(v.size() + num_tags)
        
        # Process each tag
        for i in range(num_tags):
            # Get pointer to current tag header
            ptr = <uint8_t*>data + (i * self.tagsize)
            
            # Parse fields directly from bytes
            th.code = ptr[0] | (ptr[1] << 8)
            th.datatype = ptr[2] | (ptr[3] << 8)
            th.count = (
                ptr[4] | 
                (ptr[5] << 8) | 
                (ptr[6] << 16) | 
                (ptr[7] << 24) |
                (<uint64_t>ptr[8] << 32) |
                (<uint64_t>ptr[9] << 40) |
                (<uint64_t>ptr[10] << 48) |
                (<uint64_t>ptr[11] << 56)
            )
            
            # Handle the raw value field
            th.value = (
                ptr[12] | 
                (ptr[13] << 8) | 
                (ptr[14] << 16) | 
                (ptr[15] << 24) |
                (<uint64_t>ptr[16] << 32) |
                (<uint64_t>ptr[17] << 40) |
                (<uint64_t>ptr[18] << 48) |
                (<uint64_t>ptr[19] << 56)
            )
            
            # Set interpreted offset value
            th.as_offset = th.value  # For LE, the value is already the offset for 64-bit
            
            # Initialize as_values array
            for j in range(8):
                th.as_values[j].i = 0
            
            # Convert values based on datatype
            if th.count > 0:
                if th.datatype == DATATYPE.BYTE or th.datatype == DATATYPE.ASCII or th.datatype == DATATYPE.UNDEFINED:
                    # 8-bit unsigned types
                    if th.count <= 8:
                        for j in range(min(th.count, 8)):
                            th.as_values[j].i = ptr[12+j]
                
                elif th.datatype == DATATYPE.SBYTE:
                    # 8-bit signed type
                    if th.count <= 8:
                        for j in range(min(th.count, 8)):
                            # Sign extend for negative values
                            val8 = ptr[12+j]
                            if val8 & 0x80:
                                th.as_values[j].i = -1 * ((~val8 & 0x7F) + 1)
                            else:
                                th.as_values[j].i = val8
                
                elif th.datatype == DATATYPE.SHORT:
                    # 16-bit unsigned type
                    if th.count <= 4:
                        for j in range(min(th.count, 8)):
                            if (j*2+1) < 8:  # Ensure we have enough bytes
                                th.as_values[j].i = ptr[12+j*2] | (ptr[12+j*2+1] << 8)
                
                elif th.datatype == DATATYPE.SSHORT:
                    # 16-bit signed type
                    if th.count <= 4:
                        for j in range(min(th.count, 8)):
                            if (j*2+1) < 8:
                                val16 = ptr[12+j*2] | (ptr[12+j*2+1] << 8)
                                # Sign extend
                                if val16 & 0x8000:
                                    th.as_values[j].i = -1 * ((~val16 & 0x7FFF) + 1)
                                else:
                                    th.as_values[j].i = val16
                
                elif th.datatype == DATATYPE.LONG or th.datatype == DATATYPE.IFD:
                    # 32-bit unsigned type
                    if th.count <= 2:
                        for j in range(min(th.count, 8)):
                            if (j*4+3) < 8:
                                th.as_values[j].i = (
                                    ptr[12+j*4] | 
                                    (ptr[12+j*4+1] << 8) | 
                                    (ptr[12+j*4+2] << 16) | 
                                    (ptr[12+j*4+3] << 24)
                                )
                
                elif th.datatype == DATATYPE.SLONG:
                    # 32-bit signed type
                    if th.count <= 2:
                        for j in range(min(th.count, 8)):
                            if (j*4+3) < 8:
                                val32 = (
                                    ptr[12+j*4] | 
                                    (ptr[12+j*4+1] << 8) | 
                                    (ptr[12+j*4+2] << 16) | 
                                    (ptr[12+j*4+3] << 24)
                                )
                                sval32 = <int32_t>val32
                                th.as_values[j].i = sval32  # Will automatically sign extend
                
                elif th.datatype == DATATYPE.LONG8 or th.datatype == DATATYPE.IFD8:
                    # 64-bit unsigned type
                    if th.count == 1:
                        th.as_values[0].i = th.value
                
                elif th.datatype == DATATYPE.SLONG8:
                    # 64-bit signed type
                    if th.count == 1:
                        sval64 = <int64_t>th.value
                        th.as_values[0].i = sval64  # Will automatically sign extend
                
                elif th.datatype == DATATYPE.FLOAT:
                    # 32-bit float - convert from IEEE 754 binary representation
                    if th.count <= 2:
                        for j in range(min(th.count, 2)):
                            if (j*4+3) < 8:
                                # Use a float pointer to reinterpret the bytes as float
                                float_ptr = <float*>(&ptr[12+j*4])
                                th.as_values[j].d = float_ptr[0]
                
                elif th.datatype == DATATYPE.DOUBLE:
                    # 64-bit double - convert from IEEE 754 binary representation
                    if th.count == 1:
                        # Use a double pointer to reinterpret the bytes as double
                        double_ptr = <double*>(&ptr[12])
                        th.as_values[0].d = double_ptr[0]
                
                # Other types would require data from the offset
            
            # Add to vector
            v.push_back(th)


cdef class TiffFormatBigBE(TiffFormat):
    """64-bit big-endian TIFF format."""
    def __cinit__(self) -> None:
        self.version = 43
        self.byteorder = ByteOrder.MM
        self.offsetsize = 8
        self.headersize = 16  # 2 bytes byteorder + 2 bytes version + 2 bytes bytesize + 2 bytes reserved + 8 bytes IFD offset
        self.offsetformat = '>Q'
        self.tagnosize = 8
        self.tagnoformat = '>Q'
        self.tagsize = 20
        self.tagformat1 = '>HH'
        self.tagformat2 = '>Q8s'
        self.tagoffsetthreshold = 8
        self._hash = hash((self.version, int(self.byteorder), self.offsetsize))
        
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil:
        """Parse TIFF tag headers and populate vector of TagHeader structs."""
        cdef:
            TagHeader th
            int64_t i, j, num_tags
            uint8_t* ptr
            uint8_t val8
            uint16_t val16
            int16_t sval16
            uint32_t val32
            int32_t sval32
            uint64_t val64
            int64_t sval64
            float* float_ptr
            uint32_t be_val32
            uint64_t be_val64
            double* double_ptr
            
        # Calculate number of tags
        num_tags = data_len // self.tagsize
        
        # Reserve space for efficiency
        v.reserve(v.size() + num_tags)
        
        # Process each tag
        for i in range(num_tags):
            # Get pointer to current tag header
            ptr = <uint8_t*>data + (i * self.tagsize)
            
            # Parse fields directly from bytes
            th.code = (ptr[0] << 8) | ptr[1]
            th.datatype = (ptr[2] << 8) | ptr[3]
            th.count = (
                (<uint64_t>ptr[4] << 56) |
                (<uint64_t>ptr[5] << 48) |
                (<uint64_t>ptr[6] << 40) |
                (<uint64_t>ptr[7] << 32) |
                (ptr[8] << 24) |
                (ptr[9] << 16) |
                (ptr[10] << 8) |
                ptr[11]
            )
            
            # Handle the raw value field
            th.value = (
                (<uint64_t>ptr[12] << 56) |
                (<uint64_t>ptr[13] << 48) |
                (<uint64_t>ptr[14] << 40) |
                (<uint64_t>ptr[15] << 32) |
                (ptr[16] << 24) |
                (ptr[17] << 16) |
                (ptr[18] << 8) |
                ptr[19]
            )
            
            # Set interpreted offset value
            th.as_offset = th.value  # For BE, the value is already the offset for 64-bit
            
            # Initialize as_values array
            for j in range(8):
                th.as_values[j].i = 0
            
            # Convert values based on datatype
            if th.count > 0:
                if th.datatype == DATATYPE.BYTE or th.datatype == DATATYPE.ASCII or th.datatype == DATATYPE.UNDEFINED:
                    # 8-bit unsigned types
                    if th.count <= 8:
                        for j in range(min(th.count, 8)):
                            th.as_values[j].i = ptr[12+j]
                
                elif th.datatype == DATATYPE.SBYTE:
                    # 8-bit signed type
                    if th.count <= 8:
                        for j in range(min(th.count, 8)):
                            # Sign extend for negative values
                            val8 = ptr[12+j]
                            if val8 & 0x80:
                                th.as_values[j].i = -1 * ((~val8 & 0x7F) + 1)
                            else:
                                th.as_values[j].i = val8
                
                elif th.datatype == DATATYPE.SHORT:
                    # 16-bit unsigned type
                    if th.count <= 4:
                        for j in range(min(th.count, 8)):
                            if (j*2+1) < 8:  # Ensure we have enough bytes
                                th.as_values[j].i = (ptr[12+j*2] << 8) | ptr[12+j*2+1]
                
                elif th.datatype == DATATYPE.SSHORT:
                    # 16-bit signed type
                    if th.count <= 4:
                        for j in range(min(th.count, 8)):
                            if (j*2+1) < 8:
                                val16 = (ptr[12+j*2] << 8) | ptr[12+j*2+1]
                                # Sign extend
                                if val16 & 0x8000:
                                    th.as_values[j].i = -1 * ((~val16 & 0x7FFF) + 1)
                                else:
                                    th.as_values[j].i = val16
                
                elif th.datatype == DATATYPE.LONG or th.datatype == DATATYPE.IFD:
                    # 32-bit unsigned type
                    if th.count <= 2:
                        for j in range(min(th.count, 8)):
                            if (j*4+3) < 8:
                                th.as_values[j].i = (
                                    (ptr[12+j*4] << 24) | 
                                    (ptr[12+j*4+1] << 16) | 
                                    (ptr[12+j*4+2] << 8) | 
                                    ptr[12+j*4+3]
                                )
                
                elif th.datatype == DATATYPE.SLONG:
                    # 32-bit signed type
                    if th.count <= 2:
                        for j in range(min(th.count, 8)):
                            if (j*4+3) < 8:
                                val32 = (
                                    (ptr[12+j*4] << 24) | 
                                    (ptr[12+j*4+1] << 16) | 
                                    (ptr[12+j*4+2] << 8) | 
                                    ptr[12+j*4+3]
                                )
                                sval32 = <int32_t>val32
                                th.as_values[j].i = sval32  # Will automatically sign extend
                
                elif th.datatype == DATATYPE.LONG8 or th.datatype == DATATYPE.IFD8:
                    # 64-bit unsigned type
                    if th.count == 1:
                        th.as_values[0].i = th.value
                
                elif th.datatype == DATATYPE.SLONG8:
                    # 64-bit signed type
                    if th.count == 1:
                        sval64 = <int64_t>th.value
                        th.as_values[0].i = sval64  # Will automatically sign extend
                
                elif th.datatype == DATATYPE.FLOAT:
                    # 32-bit float - convert from IEEE 754 binary representation
                    if th.count <= 2:
                        for j in range(min(th.count, 2)):
                            if (j*4+3) < 8:
                                # For big-endian, swap bytes before interpreting
                                be_val32 = (
                                    (ptr[12+j*4] << 24) | 
                                    (ptr[12+j*4+1] << 16) | 
                                    (ptr[12+j*4+2] << 8) | 
                                    ptr[12+j*4+3]
                                )
                                float_ptr = <float*>(&be_val32)
                                th.as_values[j].d = float_ptr[0]
                
                elif th.datatype == DATATYPE.DOUBLE:
                    # 64-bit double - convert from IEEE 754 binary representation
                    if th.count == 1:
                        # For big-endian, swap bytes before interpreting
                        be_val64 = (
                            (<uint64_t>ptr[12] << 56) |
                            (<uint64_t>ptr[13] << 48) |
                            (<uint64_t>ptr[14] << 40) |
                            (<uint64_t>ptr[15] << 32) |
                            (ptr[16] << 24) |
                            (ptr[17] << 16) |
                            (ptr[18] << 8) |
                            ptr[19]
                        )
                        double_ptr = <double*>(&be_val64)
                        th.as_values[0].d = double_ptr[0]
                
                # Other types would require data from the offset
            
            # Add to vector
            v.push_back(th)

cdef class TiffFormatNDPI_LE(TiffFormat):
    """32-bit little-endian TIFF format with 64-bit offsets."""
    def __cinit__(self) -> None:
        self.version = 42
        self.byteorder = ByteOrder.II
        self.offsetsize = 8
        self.headersize = 8  # 2 bytes byteorder + 2 bytes version + 4 bytes IFD offset
        self.offsetformat = '<Q'
        self.tagnosize = 2
        self.tagnoformat = '<H'
        self.tagsize = 12
        self.tagformat1 = '<HH'
        self.tagformat2 = '<I8s'
        self.tagoffsetthreshold = 4
        self._hash = hash((self.version, int(self.byteorder), self.offsetsize))
        
    cdef void parse_tag_headers(self, vector[TagHeader] &v, const uint8_t* data, int64_t data_len) noexcept nogil:
        """Parse TIFF tag headers and populate vector of TagHeader structs."""
        cdef:
            TagHeader th
            int64_t i, j, num_tags
            uint8_t* ptr
            uint8_t val8
            uint16_t val16
            int16_t sval16
            uint32_t val32
            int32_t sval32
            float* float_ptr
            
        # Calculate number of tags
        num_tags = data_len // self.tagsize
        
        # Reserve space for efficiency
        v.reserve(v.size() + num_tags)
        
        # Process each tag
        for i in range(num_tags):
            # Get pointer to current tag header
            ptr = <uint8_t*>data + (i * self.tagsize)
            
            # Parse fields directly from bytes
            th.code = ptr[0] | (ptr[1] << 8)
            th.datatype = ptr[2] | (ptr[3] << 8)
            th.count = ptr[4] | (ptr[5] << 8) | (ptr[6] << 16) | (ptr[7] << 24)
            
            # Handle the raw value field (NDPI uses 4-byte value in tag, interpreted as offset)
            th.value = ptr[8] | (ptr[9] << 8) | (ptr[10] << 16) | (ptr[11] << 24)
            
            # For NDPI, we need to handle the offset correctly elsewhere
            # (this is just the tag header parsing)
            th.as_offset = th.value
            
            # Initialize as_values array
            for j in range(8):
                th.as_values[j].i = 0
            
            # Convert values based on datatype
            if th.count > 0:
                if th.datatype == DATATYPE.BYTE or th.datatype == DATATYPE.ASCII or th.datatype == DATATYPE.UNDEFINED:
                    # 8-bit unsigned types
                    if th.count <= 4:
                        for j in range(min(th.count, 4)):
                            th.as_values[j].i = ptr[8+j]
                
                elif th.datatype == DATATYPE.SBYTE:
                    # 8-bit signed type
                    if th.count <= 4:
                        for j in range(min(th.count, 4)):
                            # Sign extend for negative values
                            val8 = ptr[8+j]
                            if val8 & 0x80:
                                th.as_values[j].i = -1 * ((~val8 & 0x7F) + 1)
                            else:
                                th.as_values[j].i = val8
                
                elif th.datatype == DATATYPE.SHORT:
                    # 16-bit unsigned type
                    if th.count <= 2:
                        for j in range(min(th.count, 2)):
                            if (j*2+1) < 4:  # Ensure we have enough bytes
                                th.as_values[j].i = ptr[8+j*2] | (ptr[8+j*2+1] << 8)
                
                elif th.datatype == DATATYPE.SSHORT:
                    # 16-bit signed type
                    if th.count <= 2:
                        for j in range(min(th.count, 2)):
                            if (j*2+1) < 4:
                                val16 = ptr[8+j*2] | (ptr[8+j*2+1] << 8)
                                # Sign extend
                                if val16 & 0x8000:
                                    th.as_values[j].i = -1 * ((~val16 & 0x7FFF) + 1)
                                else:
                                    th.as_values[j].i = val16
                
                elif th.datatype == DATATYPE.LONG or th.datatype == DATATYPE.IFD:
                    # 32-bit unsigned type
                    if th.count == 1:
                        th.as_values[0].i = th.value
                
                elif th.datatype == DATATYPE.SLONG:
                    # 32-bit signed type
                    if th.count == 1:
                        sval32 = <int32_t>th.value
                        th.as_values[0].i = sval32  # Will automatically sign extend
                
                elif th.datatype == DATATYPE.FLOAT:
                    # 32-bit float - convert from IEEE 754 binary representation
                    if th.count == 1:
                        # Use a float pointer to reinterpret the bytes as float
                        float_ptr = <float*>(&th.value)
                        th.as_values[0].d = float_ptr[0]
                
                elif th.datatype == DATATYPE.DOUBLE:
                    # 64-bit double - cannot be stored inline in NDPI's 4-byte value field
                    # Will need to read from offset
                    pass
                
                # Other types would require data from the offset
            
            # Add to vector
            v.push_back(th)


