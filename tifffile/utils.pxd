#distutils: language=c++

from libc.stdint cimport int64_t
from cpython.sequence cimport PySequence_Check

# generated with pxdgen /usr/include/c++/11/mutex -x c++

cdef extern from "<mutex>" namespace "std" nogil:
    cppclass mutex:
        mutex()
        mutex(mutex&)
        mutex& operator=(mutex&)
        void lock()
        bint try_lock()
        void unlock()
    cppclass __condvar:
        __condvar()
        __condvar(__condvar&)
        __condvar& operator=(__condvar&)
        void wait(mutex&)
        #void wait_until(mutex&, timespec&)
        #void wait_until(mutex&, clockid_t, timespec&)
        void notify_one()
        void notify_all()
    cppclass defer_lock_t:
        defer_lock_t()
    cppclass try_to_lock_t:
        try_to_lock_t()
    cppclass adopt_lock_t:
        adopt_lock_t()
    cppclass recursive_mutex:
        recursive_mutex()
        recursive_mutex(recursive_mutex&)
        recursive_mutex& operator=(recursive_mutex&)
        void lock()
        bint try_lock()
        void unlock()
    #int try_lock[_Lock1, _Lock2, _Lock3](_Lock1&, _Lock2&, _Lock3 &...)
    #void lock[_L1, _L2, _L3](_L1&, _L2&, _L3 &...)
    cppclass lock_guard[_Mutex]:
        ctypedef _Mutex mutex_type
        lock_guard(mutex_type&)
        lock_guard(mutex_type&, adopt_lock_t)
        lock_guard(lock_guard&)
        lock_guard& operator=(lock_guard&)
    cppclass scoped_lock[_MutexTypes]:
        #scoped_lock(_MutexTypes &..., ...)
        scoped_lock()
        scoped_lock(_MutexTypes &)
        #scoped_lock(adopt_lock_t, _MutexTypes &...)
        #scoped_lock(scoped_lock&)
        scoped_lock& operator=(scoped_lock&)
    cppclass unique_lock[_Mutex]:
        ctypedef _Mutex mutex_type
        unique_lock()
        unique_lock(mutex_type&)
        unique_lock(mutex_type&, defer_lock_t)
        unique_lock(mutex_type&, try_to_lock_t)
        unique_lock(mutex_type&, adopt_lock_t)
        unique_lock(unique_lock&)
        unique_lock& operator=(unique_lock&)
        #unique_lock(unique_lock&&)
        #unique_lock& operator=(unique_lock&&)
        void lock()
        bint try_lock()
        void unlock()
        void swap(unique_lock&)
        mutex_type* release()
        bint owns_lock()
        mutex_type* mutex()
    void swap[_Mutex](unique_lock[_Mutex]&, unique_lock[_Mutex]&)

cdef void lock_gil_friendly_block(unique_lock[recursive_mutex] &m) noexcept

cdef inline void lock_gil_friendly(unique_lock[recursive_mutex] &m,
                                   recursive_mutex &mutex) noexcept:
    """
    Must be called to lock our mutexes whenever we hold the gil
    """
    m = unique_lock[recursive_mutex](mutex, defer_lock_t())
    # Fast path which will be hit almost always
    if m.try_lock():
        return
    # Slow path
    lock_gil_friendly_block(m)

cdef inline int64_t product(sequence):
    if not PySequence_Check(sequence):
        return <int64_t>sequence
    cdef int64_t r = 1
    for e in sequence:
        r *= int(e)

cdef inline str bytes2str_stripnull(bytes b):
    """Optimized version of byte2str(stripnull(b, first=False))."""
    cdef int i = len(b)
    cdef unsigned char* p = <unsigned char*>b
    with nogil:
        while i:
            i -= 1
            if p[i]:
                break
    if i == len(b) - 1:
        return b.decode('utf-8')
    return b[: i + 1].decode('utf-8')

cdef inline str bytes2str_stripnull_last(bytes b):
    """Optimized version of 
    bytes2str(stripnull(b, first=False).strip()."""
    cdef int size = len(b)
    cdef unsigned char* p = <unsigned char*>b
    cdef int i = 0
    cdef int start, end
    with nogil:
        while i < size:
            # omit 0 and whitespace
            if p[i] or p[i] == 32:
                break
            i += 1
        start = i
        i = size
        while i:
            i -= 1
            if p[i] or p[i] == 32:
                break
        end = i + 1
    if start == 0 and end == size:
        return b.decode('utf-8')
    return b[start:end].decode('utf-8')