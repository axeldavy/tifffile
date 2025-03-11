#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=True
#distutils: language=c++

from libc.stdint cimport int64_t, uint16_t, uint32_t, int64_t, uint64_t
import struct
from cpython.bytes cimport PyBytes_AS_STRING

from .utils import indent

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
        
    cpdef tuple parse_tag_header(self, bytes header):
        """Parse TIFF tag header and return code, dtype, count, and value.
        
        Parameters
        ----------
        header : bytes
            TIFF tag header bytes.
            
        Returns
        -------
        tuple
            (code, dtype, count, value)
        """
        cdef:
            int code
            int dtype
            int64_t count
            bytes value
            
        # Fallback implementation using struct.unpack
        code, dtype = struct.unpack(self.tagformat1, header[:4])
        count, value = struct.unpack(self.tagformat2, header[4:])
        return (code, dtype, count, value)
    
    cpdef uint64_t interprete_offset(self, bytes value):
        """Convert bytes value from tag header to offset.
        
        Parameters
        ----------
        value : bytes
            Value field from tag header
            
        Returns
        -------
        uint64_t
            Offset value
            
        Raises
        ------
        ValueError
            If the value bytes cannot be interpreted as an offset
        """
        # Fallback implementation using struct.unpack
        try:
            # Use first 'offsetsize' bytes of the value
            return struct.unpack(self.offsetformat, value[:self.offsetsize])[0]
        except struct.error:
            raise ValueError(f"Cannot convert {value!r} to an offset")

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
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(header)
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
        self.offsetformat = '<I'
        self.tagnosize = 2
        self.tagnoformat = '<H'
        self.tagsize = 12
        self.tagformat1 = '<HH'
        self.tagformat2 = '<I4s'
        self.tagoffsetthreshold = 4
        self._hash = hash((self.version, int(self.byteorder), self.offsetsize))
        
    cpdef tuple parse_tag_header(self, bytes header):
        """Parse TIFF tag header and return code, dtype, count, and value."""
        cdef:
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(header)
            uint16_t code = data[0] | (data[1] << 8)
            uint16_t dtype = data[2] | (data[3] << 8)
            uint32_t count = data[4] | (data[5] << 8) | (data[6] << 16) | (data[7] << 24)
            bytes value = header[8:12]
            
        return (code, dtype, count, value)
    
    cpdef uint64_t interprete_offset(self, bytes value):
        """Convert bytes value from tag header to offset."""
        cdef:
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(value)
            uint32_t offset = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)
            
        return offset

cdef class TiffFormatClassicBE(TiffFormat):
    """32-bit big-endian TIFF format."""
    def __cinit__(self) -> None:
        self.version = 42
        self.byteorder = ByteOrder.MM
        self.offsetsize = 4
        self.offsetformat = '>I'
        self.tagnosize = 2
        self.tagnoformat = '>H'
        self.tagsize = 12
        self.tagformat1 = '>HH'
        self.tagformat2 = '>I4s'
        self.tagoffsetthreshold = 4
        self._hash = hash((self.version, int(self.byteorder), self.offsetsize))
        
    cpdef tuple parse_tag_header(self, bytes header):
        """Parse TIFF tag header and return code, dtype, count, and value."""
        cdef:
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(header)
            uint16_t code = (data[0] << 8) | data[1]
            uint16_t dtype = (data[2] << 8) | data[3]
            uint32_t count = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
            bytes value = header[8:12]
            
        return (code, dtype, count, value)
    
    cpdef uint64_t interprete_offset(self, bytes value):
        """Convert bytes value from tag header to offset."""
        cdef:
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(value)
            uint32_t offset = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]
            
        return offset

cdef class TiffFormatBigLE(TiffFormat):
    """64-bit little-endian TIFF format."""
    def __cinit__(self) -> None:
        self.version = 43
        self.byteorder = ByteOrder.II
        self.offsetsize = 8
        self.offsetformat = '<Q'
        self.tagnosize = 8
        self.tagnoformat = '<Q'
        self.tagsize = 20
        self.tagformat1 = '<HH'
        self.tagformat2 = '<Q8s'
        self.tagoffsetthreshold = 8
        self._hash = hash((self.version, int(self.byteorder), self.offsetsize))
        
    cpdef tuple parse_tag_header(self, bytes header):
        """Parse TIFF tag header and return code, dtype, count, and value."""
        cdef:
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(header)
            uint16_t code = data[0] | (data[1] << 8)
            uint16_t dtype = data[2] | (data[3] << 8)
            uint64_t count = (
                data[4] | 
                (data[5] << 8) | 
                (data[6] << 16) | 
                (data[7] << 24) |
                (<uint64_t>data[8] << 32) |
                (<uint64_t>data[9] << 40) |
                (<uint64_t>data[10] << 48) |
                (<uint64_t>data[11] << 56)
            )
            bytes value = header[12:20]
            
        return (code, dtype, count, value)
    
    cpdef uint64_t interprete_offset(self, bytes value):
        """Convert bytes value from tag header to offset."""
        cdef:
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(value)
            uint64_t offset = (
                data[0] | 
                (data[1] << 8) | 
                (data[2] << 16) | 
                (data[3] << 24) |
                (<uint64_t>data[4] << 32) |
                (<uint64_t>data[5] << 40) |
                (<uint64_t>data[6] << 48) |
                (<uint64_t>data[7] << 56)
            )
            
        return offset

cdef class TiffFormatBigBE(TiffFormat):
    """64-bit big-endian TIFF format."""
    def __cinit__(self) -> None:
        self.version = 43
        self.byteorder = ByteOrder.MM
        self.offsetsize = 8
        self.offsetformat = '>Q'
        self.tagnosize = 8
        self.tagnoformat = '>Q'
        self.tagsize = 20
        self.tagformat1 = '>HH'
        self.tagformat2 = '>Q8s'
        self.tagoffsetthreshold = 8
        self._hash = hash((self.version, int(self.byteorder), self.offsetsize))
        
    cpdef tuple parse_tag_header(self, bytes header):
        """Parse TIFF tag header and return code, dtype, count, and value."""
        cdef:
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(header)
            uint16_t code = (data[0] << 8) | data[1]
            uint16_t dtype = (data[2] << 8) | data[3]
            uint64_t count = (
                (<uint64_t>data[4] << 56) |
                (<uint64_t>data[5] << 48) |
                (<uint64_t>data[6] << 40) |
                (<uint64_t>data[7] << 32) |
                (data[8] << 24) |
                (data[9] << 16) |
                (data[10] << 8) |
                data[11]
            )
            bytes value = header[12:20]
            
        return (code, dtype, count, value)
    
    cpdef uint64_t interprete_offset(self, bytes value):
        """Convert bytes value from tag header to offset."""
        cdef:
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(value)
            uint64_t offset = (
                (<uint64_t>data[0] << 56) |
                (<uint64_t>data[1] << 48) |
                (<uint64_t>data[2] << 40) |
                (<uint64_t>data[3] << 32) |
                (data[4] << 24) |
                (data[5] << 16) |
                (data[6] << 8) |
                data[7]
            )
            
        return offset

cdef class TiffFormatNDPI_LE(TiffFormat):
    """32-bit little-endian TIFF format with 64-bit offsets."""
    def __cinit__(self) -> None:
        self.version = 42
        self.byteorder = ByteOrder.II
        self.offsetsize = 8
        self.offsetformat = '<Q'
        self.tagnosize = 2
        self.tagnoformat = '<H'
        self.tagsize = 12
        self.tagformat1 = '<HH'
        self.tagformat2 = '<I8s'
        self.tagoffsetthreshold = 4
        self._hash = hash((self.version, int(self.byteorder), self.offsetsize))
        
    cpdef tuple parse_tag_header(self, bytes header):
        """Parse TIFF tag header and return code, dtype, count, and value."""
        cdef:
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(header)
            uint16_t code = data[0] | (data[1] << 8)
            uint16_t dtype = data[2] | (data[3] << 8)
            uint32_t count = data[4] | (data[5] << 8) | (data[6] << 16) | (data[7] << 24)
            bytes value = header[8:16]  # NDPI uses 8-byte value field

        return (code, dtype, count, value)
    
    cpdef uint64_t interprete_offset(self, bytes value):
        """Convert bytes value from tag header to offset."""
        cdef:
            const unsigned char* data = <const unsigned char*>PyBytes_AS_STRING(value)
            uint64_t offset = (
                data[0] | 
                (data[1] << 8) | 
                (data[2] << 16) | 
                (data[3] << 24) |
                (<uint64_t>data[4] << 32) |
                (<uint64_t>data[5] << 40) |
                (<uint64_t>data[6] << 48) |
                (<uint64_t>data[7] << 56)
            )
            
        return offset

