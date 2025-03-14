# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=True
# cython: cdivision=True
# cython: nonecheck=False
#distutils: language=c++

from libc.stdint cimport int64_t
from libcpp.vector cimport vector

from .files cimport FileHandle
from .pages cimport TiffPage

cdef class TiffPages:
    """Sequence of TIFF image file directories (IFD chain).

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

    def __init__(self):
        raise ValueError("Must be created from TiffFile, TiffPage, or TiffFrame")

    @staticmethod
    cdef TiffPages from_file(
        FileHandle filehandle,
        TiffFormat tiff,
        int64_t offset
    ):
        """Create TiffPages from file."""
        cdef TiffPages pages = TiffPages.__new__(TiffPages)
        pages.filehandle = filehandle
        pages.tiff = tiff
        pages._page_offsets.push_back(offset)
        pages._pages = [None] # Lazy loading
        return pages

    @staticmethod
    cdef TiffPages from_parent(
        FileHandle filehandle,
        TiffFormat tiff,
        vector[int64_t] offsets
    ):
        """Create TiffPages from parent."""
        cdef TiffPages pages = TiffPages.__new__(TiffPages)
        pages.filehandle = filehandle
        pages.tiff = tiff
        pages._page_offsets = offsets
        pages._pages = [None] * len(offsets) # Lazy loading
        return pages

    def __len__(self):
        """Return number of pages in file."""
        return self._page_offsets.size()

    def __getitem__(self, int64_t key):
        """Return specified page from cache or file."""
        return self.getpage(key)

    cdef TiffPage getpage(self, int64_t key):
        """Return specified page from the pages array"""
        cdef TiffPage page
        if key < 0 or key >= self._page_offsets.size():
            raise IndexError(f'index {key} out of range({self._page_offsets.size()})')
        if self._pages[key] is not None:
            return self._pages[key]
        # Page is not loaded and must be loaded now
        page = TiffPage.from_file(self.filehandle, self.tiff, self._page_offsets[key])
        assert page is not None
        self._pages[key] = page
        return page

