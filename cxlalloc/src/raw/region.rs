use core::ffi;
use core::fmt::Display;
use core::fmt::Write as _;
use core::num::NonZeroUsize;
use core::ops::Deref;
use core::ptr;
use core::ptr::NonNull;
use std::io;

use arrayvec::ArrayString;

use crate::raw::backend;
use crate::raw::backend::Backend;

use super::Csr;

#[repr(C, align(4096))]
pub(crate) struct Page([u8; 4096]);

#[derive(Clone, Debug)]
pub(crate) struct Id(ArrayString<{ Self::SIZE }>);

impl Id {
    pub(crate) const SIZE: usize = 32;

    pub(crate) fn new(inner: &str) -> Self {
        ArrayString::from(inner)
            .map(Self)
            .expect("Region identifiers must be less than 32 bytes")
    }

    pub(crate) fn with_suffix<T: Display>(&self, suffix: T) -> Self {
        let mut id = self.clone().0;
        write!(&mut id, "-{}", suffix).unwrap();
        Self(id)
    }
}

impl Deref for Id {
    type Target = ArrayString<{ Self::SIZE }>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub(crate) struct Reservation(NonNull<Page>);

impl Reservation {
    pub(crate) const SIZE: NonZeroUsize = NonZeroUsize::new(1 << 40).unwrap();

    // In order to keep heap regions contiguous when extending, we need
    // to reserve an unbacked region of virtual address space,
    // and then overwrite it later via `mmap` with `MMAP_FIXED`.
    pub(crate) fn new() -> crate::Result<Self> {
        let address = Self::mmap(Self::SIZE)?;
        Ok(Self(address))
    }

    pub(crate) fn new_contiguous<const COUNT: usize>() -> crate::Result<[Self; COUNT]> {
        let total = NonZeroUsize::new(Self::SIZE.get() * COUNT).unwrap();
        let address = Self::mmap(total)?;
        Ok(std::array::from_fn(|i| {
            Self(unsafe { address.byte_add(Self::SIZE.get() * i) })
        }))
    }

    fn mmap(size: NonZeroUsize) -> crate::Result<NonNull<Page>> {
        match unsafe {
            libc::mmap64(
                ptr::null_mut(),
                size.get(),
                libc::PROT_NONE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                -1,
                0,
            )
        } {
            libc::MAP_FAILED => Err(crate::Error::Mmap(io::Error::last_os_error())),
            actual => Ok(NonNull::new(actual).unwrap().cast::<Page>()),
        }
    }

    fn munmap(&self) -> crate::Result<()> {
        unsafe { munmap(self.0, Self::SIZE) }
    }

    fn start(&self) -> NonNull<Page> {
        self.0.cast()
    }

    fn end(&self) -> NonNull<Page> {
        unsafe { self.0.byte_add(Self::SIZE.get()) }
    }
}

pub(crate) trait Region {
    fn address(&self) -> NonNull<Page>;
    fn is_clean(&self) -> bool;
    fn id(&self) -> &str;
    fn unmap(&self) -> crate::Result<()>;
}

pub(crate) struct Fixed {
    pub(super) id: Id,
    pub(super) clean: bool,
    pub(super) address: NonNull<Page>,
    pub(super) size: NonZeroUsize,
}

pub(crate) struct Sequential {
    id: Id,
    clean: bool,
    reservation: Reservation,
    size: NonZeroUsize,
}

pub(crate) struct Random {
    id: Id,
    reservation: Reservation,
}

impl Fixed {
    pub(super) fn new(backend: &Backend, id: Id, size: NonZeroUsize) -> crate::Result<Self> {
        let size = NonZeroUsize::new(size.get().next_multiple_of(crate::SIZE_PAGE)).unwrap();
        let file = backend.allocate(id.clone(), size)?;
        let address = unsafe {
            mmap(
                None,
                size,
                backend.numa().cloned(),
                backend.populate(),
                &file,
            )?
        };
        Ok(Self {
            id,
            address,
            clean: file.clean,
            size,
        })
    }
}

impl Region for Fixed {
    fn address(&self) -> NonNull<Page> {
        self.address
    }

    fn is_clean(&self) -> bool {
        self.clean
    }

    fn id(&self) -> &str {
        &self.id.0
    }

    fn unmap(&self) -> crate::Result<()> {
        unsafe { munmap(self.address, self.size) }
    }
}

impl Sequential {
    pub(super) fn new(
        backend: &Backend,
        id: Id,
        reservation: Reservation,
        size: NonZeroUsize,
        lazy: bool,
    ) -> crate::Result<Self> {
        let size = NonZeroUsize::new(size.get().next_multiple_of(crate::SIZE_PAGE)).unwrap();

        let clean = match lazy {
            true => false,
            false => {
                let file = backend.allocate(id.with_suffix(0), size)?;
                unsafe {
                    mmap(
                        Some(reservation.0),
                        size,
                        backend.numa().cloned(),
                        backend.populate(),
                        &file,
                    )
                }?;
                file.clean
            }
        };

        Ok(Sequential {
            id,
            clean,
            reservation,
            size,
        })
    }

    pub(crate) fn map(&self, backend: &Backend, offset: usize) -> crate::Result<()> {
        let index = offset / self.size.get();
        let file = backend.allocate(self.id.with_suffix(index), self.size)?;

        unsafe {
            mmap(
                Some(self.reservation.0.byte_add(self.size.get() * index)),
                self.size,
                backend.numa().cloned(),
                backend.populate(),
                &file,
            )
        }?;

        Ok(())
    }
}

impl Region for Sequential {
    fn address(&self) -> NonNull<Page> {
        self.reservation.0
    }

    fn is_clean(&self) -> bool {
        self.clean
    }

    fn id(&self) -> &str {
        &self.id.0
    }

    /// Remove all virtual address space mappings for this region.
    fn unmap(&self) -> crate::Result<()> {
        self.reservation.munmap()
    }
}

impl Random {
    pub(super) fn new(id: Id, reservation: Reservation) -> io::Result<Self> {
        Ok(Random { id, reservation })
    }

    pub(crate) fn contains(&self, pointer: NonNull<ffi::c_void>) -> bool {
        (self.reservation.start().cast()..self.reservation.end().cast()).contains(&pointer)
    }

    pub(crate) fn map(
        &self,
        backend: &Backend,
        offset: usize,
        size: NonZeroUsize,
    ) -> crate::Result<()> {
        let file = backend.allocate(self.id.with_suffix(format_args!("{:#x}", offset)), size)?;
        unsafe {
            mmap(
                Some(self.address().byte_add(offset)),
                size,
                backend.numa().cloned(),
                backend.populate(),
                &file,
            )
        }?;

        Ok(())
    }

    pub(crate) fn unmap(&self, backend: &Backend, offset: usize, size: NonZeroUsize) {
        let id = self.id.with_suffix(format_args!("{:#x}", offset));
        let _ = unsafe { munmap(self.address().byte_add(offset), size) };
        let _ = backend.unlink(&id);
    }
}

impl Region for Random {
    fn address(&self) -> NonNull<Page> {
        self.reservation.start()
    }

    fn is_clean(&self) -> bool {
        false
    }

    fn id(&self) -> &str {
        &self.id.0
    }

    fn unmap(&self) -> crate::Result<()> {
        self.reservation.munmap()
    }
}

unsafe fn munmap(address: NonNull<Page>, size: NonZeroUsize) -> crate::Result<()> {
    match unsafe { libc::munmap(address.as_ptr().cast(), size.get()) } {
        0 => Ok(()),
        _ => Err(crate::Error::Munmap(io::Error::last_os_error())),
    }
}

unsafe fn mmap(
    address: Option<NonNull<Page>>,
    size: NonZeroUsize,
    numa: Option<::shm::Numa>,
    populate: Option<::shm::Populate>,
    file: &backend::File,
) -> crate::Result<NonNull<Page>> {
    let actual = match libc::mmap64(
        address
            .map(NonNull::as_ptr)
            .unwrap_or_else(ptr::null_mut)
            .cast(),
        size.get(),
        libc::PROT_READ | libc::PROT_WRITE,
        file.flags()
            | address.map(|_| libc::MAP_FIXED).unwrap_or(0)
            | if matches!(populate, Some(::shm::Populate::PageTable)) {
                libc::MAP_POPULATE
            } else {
                0
            },
        file.fd(),
        file.offset,
    ) {
        libc::MAP_FAILED => return Err(crate::Error::Mmap(io::Error::last_os_error())),
        actual => NonNull::new(actual).unwrap().cast::<Page>(),
    };

    if let Some(expected) = address {
        assert_eq!(expected, actual);
    }

    if let Some(numa) = numa {
        unsafe {
            ::shm::Raw::mbind(numa, actual.as_ptr().cast(), size.get())
                .map_err(crate::Error::Mbind)?;
        }
    }

    if matches!(populate, Some(::shm::Populate::Physical)) {
        unsafe {
            ::shm::Raw::madvise(actual.as_ptr().cast(), size.get())
                .map_err(crate::Error::Madvise)?;
        }
    }

    Ok(actual)
}
