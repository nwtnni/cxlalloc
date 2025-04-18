pub mod backend;
pub(crate) mod region;

pub use backend::Backend;

pub use raw_builder::State as BuilderState;
pub(crate) use region::Page;
use region::Region;
pub(crate) use region::Reservation;
pub use RawBuilder as Builder;

use core::alloc::Layout;
use core::cell::UnsafeCell;
use core::ffi;
use core::ffi::CStr;
use core::num::NonZeroUsize;
use core::ptr;
use core::ptr::NonNull;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;
use std::io;
use std::os::fd::AsRawFd as _;
use std::os::fd::FromRawFd as _;
use std::os::fd::OwnedFd;
use std::sync::OnceLock;

use bon::bon;

use crate::allocator;
use crate::heap;
use crate::huge;
use crate::size;
use crate::size::Bracket;
use crate::slab;
use crate::stat;
use crate::thread;
use crate::view;
use crate::Allocator;
use crate::Data;
use crate::Heap;
use crate::Huge;
use crate::Slab;
use crate::BATCH_BUMP_POP;
use crate::BATCH_GLOBAL_PUSH;
use crate::COUNT_CACHE_SLAB;

/// This type represents sole ownership of an initialized backing store
/// for the heap.
pub struct Raw {
    pub(crate) backend: Backend,

    // - Global persistent root: 1
    // - Help array: # threads
    // - Small and large heaps
    //   - Global stack: 1
    //   - Bump pointer: 1
    // - Huge heap
    //   - Next slot: 1
    //   - Slot array: # huge allocations (extend)
    pub(crate) shared: region::Fixed,

    // - Local persistent roots: # threads
    // - Small and large heaps
    //   - Unsized free list: # threads
    //   - Sized free lists: # sizes * # threads
    // - Huge heap
    //   - Descriptor lists: # threads
    pub(crate) owned: region::Fixed,

    // Slab metadata regions
    pub(crate) local_small: region::Sequential,
    pub(crate) local_large: region::Sequential,
    pub(crate) remote_small: region::Sequential,
    pub(crate) remote_large: region::Sequential,

    // Data regions, must be contiguous
    pub(crate) data_small: region::Sequential,
    pub(crate) data_large: region::Sequential,
    pub(crate) data_huge: region::Random,

    stat: stat::process::Recorder,

    /// Free on drop
    free: bool,
}

/// # Safety
///
/// The memory regions are mapped for the entire process, so
/// the pointers remain valid when transferred to a different thread.
unsafe impl Send for Raw {}

/// # Safety
///
/// The only (public) way to interact with a [`Raw`] is through
/// a [`crate::Heap`] or [`crate::Allocator`], which expose
/// thread-safe methods.
unsafe impl Sync for Raw {}

/// Compute size and offsets for a sequence of types in memory.
macro_rules! layout {
    ($head:ty $(, $tail:ty)* $(,)?) => {
        {
            let mut offsets = vec![0];
            let mut layout = Layout::new::<$head>();
            for field in [$(Layout::new::<$tail>()),*] {
                let (next, offset) = layout.extend(field).unwrap();
                layout = next;
                offsets.push(offset);
            }
            (NonZeroUsize::new(layout.pad_to_align().size()).unwrap(), offsets)
        }
    };
}

pub(crate) static MCAS: OnceLock<Mcas> = OnceLock::new();
pub(crate) static TARGET: OnceLock<Buffer> = OnceLock::new();
thread_local! {
    pub(crate) static THREAD_ID: AtomicU64 = const { AtomicU64::new(0) };
}

#[bon]
impl Raw {
    #[builder]
    pub fn new(
        #[builder(finish_fn)] id: &str,
        #[builder(default, into)] backend: Backend,
        #[builder(default)] size_small: usize,
        #[builder(default)] size_large: usize,
        #[builder(default = 1)] thread_count: usize,
        #[builder(default)] free: bool,
        cache_local: Option<usize>,
        batch_global: Option<usize>,
        batch_bump: Option<usize>,
    ) -> crate::Result<Raw> {
        log::info!(
            "Requesting heap with \
            backend = {}, \
            size_small = {}, \
            size_large = {}, \
            thread_count = {}",
            backend.name(),
            size_small,
            size_large,
            thread_count,
        );

        if let Some(cache_local) = cache_local {
            COUNT_CACHE_SLAB.store(cache_local, Ordering::Relaxed);
        }

        if let Some(batch_global) = batch_global {
            BATCH_GLOBAL_PUSH.store(batch_global, Ordering::Relaxed);
        }

        if let Some(batch_bump) = batch_bump {
            BATCH_BUMP_POP.store(batch_bump, Ordering::Relaxed);
        }

        let id = region::Id::new(id);

        let (_shared_size, _) = Self::shared();
        let mut csr = Csr::new().unwrap();

        MCAS.get_or_init(|| Mcas::new(&mut csr).unwrap());
        let target = Buffer::target(&mut csr).unwrap();
        TARGET.get_or_init(|| target);

        // FIXME: support extension for huge allocation region?
        let shared = region::Fixed {
            id: region::Id::new("shared"),
            address: NonNull::new(target.address_virt).unwrap().cast(),
            clean: true,
            size: NonZeroUsize::new(PAGE_SIZE * 16).unwrap(),
        };

        let (owned_size, _) = Self::owned();
        let owned = region::Fixed::new(&backend, id.with_suffix("owned"), owned_size)?;

        let (small_lazy, small) = match NonZeroUsize::new(
            size_small.next_multiple_of(size::Small::SIZE_SLAB) / size::Small::SIZE_SLAB,
        )
        .map(|count| Heap::<view::Unfocus, size::Small>::layout(count).unwrap())
        {
            None => (true, Default::default()),
            Some(layout) => (false, layout),
        };

        let local_small_reservation = Reservation::new()?;
        let local_small = region::Sequential::new(
            &backend,
            id.with_suffix("ls"),
            local_small_reservation,
            small.locals,
            small_lazy,
        )?;

        let remote_small_reservation = Reservation::new()?;
        let remote_small = region::Sequential::new(
            &backend,
            id.with_suffix("rs"),
            remote_small_reservation,
            small.remotes,
            small_lazy,
        )?;

        let (large_lazy, large) = match NonZeroUsize::new(
            size_large.next_multiple_of(size::Large::SIZE_SLAB) / size::Large::SIZE_SLAB,
        )
        .map(|count| Heap::<view::Unfocus, size::Large>::layout(count).unwrap())
        {
            None => (true, Default::default()),
            Some(layout) => (false, layout),
        };

        let local_large_reservation = Reservation::new()?;
        let local_large = region::Sequential::new(
            &backend,
            id.with_suffix("ll"),
            local_large_reservation,
            large.locals,
            large_lazy,
        )?;

        let remote_large_reservation = Reservation::new()?;
        let remote_large = region::Sequential::new(
            &backend,
            id.with_suffix("rl"),
            remote_large_reservation,
            large.remotes,
            large_lazy,
        )?;

        let [data_small_reservation, data_large_reservation, data_huge_reservation] =
            Reservation::new_contiguous()?;

        let data_small = region::Sequential::new(
            &backend,
            id.with_suffix("ds"),
            data_small_reservation,
            small.data,
            small_lazy,
        )?;

        let data_large = region::Sequential::new(
            &backend,
            id.with_suffix("dl"),
            data_large_reservation,
            large.data,
            large_lazy,
        )?;

        let data_huge = region::Random::new(id.with_suffix("dh"), data_huge_reservation)?;

        Ok(Self {
            backend,
            shared,
            owned,
            local_small,
            local_large,
            remote_small,
            remote_large,
            data_small,
            data_large,
            data_huge,
            stat: stat::process::Recorder::default(),
            free,
        })
    }
}

impl Raw {
    pub fn allocator<S, O>(&self, id: thread::Id) -> Allocator<S, O> {
        THREAD_ID.with(|thread_id| thread_id.store(u16::from(id) as u64, Ordering::Relaxed));
        unsafe { Allocator::new(self.unfocused().focus(id)) }
    }

    pub fn report(&self) -> impl Iterator<Item = stat::Report> + '_ {
        self.stat.report()
    }

    pub fn map(&self, id: thread::Id, address: *mut ffi::c_void) -> bool {
        let Some(address) = NonNull::new(address) else {
            return false;
        };

        let allocator = self.unfocused::<(), ()>();

        let context = crate::allocator::Context {
            id,
            help: &allocator.shared.help,
            log: &mut None,
        };

        match allocator.small.try_map(
            &self.backend,
            &self.local_small,
            &self.remote_small,
            &self.data_small,
            &context,
            address,
        ) {
            Ok(()) => {
                self.stat.record(stat::process::Event::FaultSmall);
                return true;
            }
            Err(crate::Error::OutOfBounds) => (),
            Err(error) => panic!("Failed to extend small heap at {:x?}: {}", address, error),
        }

        match allocator.large.try_map(
            &self.backend,
            &self.local_large,
            &self.remote_large,
            &self.data_large,
            &context,
            address,
        ) {
            Ok(()) => {
                self.stat.record(stat::process::Event::FaultLarge);
                return true;
            }
            Err(crate::Error::OutOfBounds) => (),
            Err(error) => panic!("Failed to extend large heap at {:x?}: {}", address, error),
        }

        match allocator.huge.try_map(&allocator.small.data, id, address) {
            Ok(()) => {
                self.stat.record(stat::process::Event::FaultHuge);
                return true;
            }
            Err(crate::Error::OutOfBounds) => (),
            Err(error) => panic!("Failed to map huge allocation at {:x?}: {}", address, error),
        }

        false
    }

    fn unfocused<S, O>(&self) -> allocator::Allocator<view::Unfocus, S, O> {
        let (_, shared_offsets) = Self::shared();
        let (_, owned_offsets) = Self::owned();
        let shared = self.shared.address().as_ptr();
        let owned = self.owned.address().as_ptr();
        unsafe {
            // Note: calls layout code at runtime. Ideally the layout information could be
            // a const, but some APIs (Layout::extend, Layout::pad_to_align) aren't
            // const yet.
            allocator::Allocator::new(
                (),
                shared
                    .wrapping_byte_add(shared_offsets[0])
                    .cast::<allocator::Shared<S>>()
                    .as_ref()
                    .unwrap(),
                owned
                    .wrapping_byte_add(owned_offsets[0])
                    .cast::<thread::Array<UnsafeCell<allocator::Owned<O>>>>()
                    .as_ref()
                    .unwrap(),
                Heap::<view::Unfocus, size::Small>::new(
                    shared
                        .wrapping_byte_add(shared_offsets[1])
                        .cast::<heap::Shared<size::Small>>()
                        .as_ref()
                        .unwrap(),
                    owned
                        .wrapping_byte_add(owned_offsets[1])
                        .cast::<thread::Array<UnsafeCell<heap::Owned<size::Small>>>>()
                        .as_ref()
                        .unwrap(),
                    Slab::new(
                        slab::Slice::from_raw(self.local_small.address().cast()),
                        slab::Slice::from_raw(self.remote_small.address().cast()),
                    ),
                    Data::<size::Small>::new(self.data_small.address()),
                ),
                Heap::<view::Unfocus, size::Large>::new(
                    shared
                        .wrapping_byte_add(shared_offsets[2])
                        .cast::<heap::Shared<size::Large>>()
                        .as_ref()
                        .unwrap(),
                    owned
                        .wrapping_byte_add(owned_offsets[2])
                        .cast::<thread::Array<UnsafeCell<heap::Owned<size::Large>>>>()
                        .as_ref()
                        .unwrap(),
                    Slab::new(
                        slab::Slice::from_raw(self.local_large.address().cast()),
                        slab::Slice::from_raw(self.remote_large.address().cast()),
                    ),
                    Data::<size::Large>::new(self.data_large.address()),
                ),
                Huge::new(
                    &self.backend,
                    &self.data_huge,
                    shared
                        .wrapping_byte_add(shared_offsets[3])
                        .cast::<huge::Shared>()
                        .as_ref()
                        .unwrap(),
                    owned
                        .wrapping_byte_add(owned_offsets[3])
                        .cast::<thread::Array<huge::Owned>>()
                        .as_ref()
                        .unwrap(),
                    Data::<size::Huge>::new(self.data_huge.address()),
                ),
            )
        }
    }

    pub fn is_clean(&self) -> bool {
        self.regions().any(Region::is_clean)
    }

    pub(crate) fn shared() -> (NonZeroUsize, Vec<usize>) {
        layout!(
            allocator::Shared<()>,
            heap::Shared<size::Small>,
            heap::Shared<size::Large>,
            huge::Shared,
        )
    }

    pub(crate) fn owned() -> (NonZeroUsize, Vec<usize>) {
        layout!(
            thread::Array<UnsafeCell<allocator::Owned<()>>>,
            thread::Array<UnsafeCell<heap::Owned<size::Small>>>,
            thread::Array<UnsafeCell<heap::Owned<size::Large>>>,
            thread::Array<huge::Owned>,
        )
    }

    fn regions(&self) -> impl Iterator<Item = &dyn Region> {
        [
            &self.shared as &dyn Region,
            &self.owned,
            &self.local_small,
            &self.local_large,
            &self.remote_small,
            &self.remote_large,
            &self.data_small,
            &self.data_large,
            &self.data_huge,
        ]
        .into_iter()
    }
}

impl Drop for Raw {
    fn drop(&mut self) {
        self.regions().for_each(|region| match region.unmap() {
            Ok(()) => (),
            Err(error) => log::error!("Failed to unmap {} region: {:?}", region.id(), error),
        });

        if !self.free {
            return;
        }

        todo!()
    }
}

pub fn mcas(address: *mut u64, old: u64, new: u64) -> Result<u64, u64> {
    let mcas = MCAS.get().unwrap();
    let target = TARGET.get().unwrap();
    let phys = target.translate(address);
    let id = THREAD_ID.with(|id| id.load(Ordering::Relaxed));

    log::warn!(
        "{} {:?} {:?} mcas: v{:x?} p{:x?} o{} n{}",
        id,
        mcas,
        target,
        address,
        phys,
        old,
        new
    );

    let wr = mcas.write.address_virt.cast::<u64>();
    let rd = mcas.read.address_virt.cast::<u64>();

    unsafe {
        let mut buffer: Aligned = Aligned([old, new, phys, id * 2, 0, 0, 0, 0]);

        core::arch::asm! {
            "movdir64b 0x0({dest}), {src}",
            dest = in(reg) wr,
            src  = in(reg) &mut buffer as *mut _,
        }

        // wr.write_volatile(old);
        // wr.add(1).write_volatile(new);
        // wr.add(2).write_volatile(phys);
        // wr.add(3).write_volatile(id * 2);

        // core::arch::x86_64::_mm_clflush(wr.cast());
        core::arch::x86_64::_mm_clflush(rd.cast());
        core::arch::x86_64::_mm_mfence();

        let rd = rd.byte_add(id as usize * 64);
        let mut out = [0u64; 2];

        core::arch::asm! {
            "movdqu xmm0, [{input}]",
            "movdqu [{output}], xmm0",
            input = in(reg) rd,
            output = in(reg) out.as_ptr(),
        }

        let result = out[0];
        let success = out[1];

        log::warn!("{id} mcas result: {result} {success}");

        match success {
            0 => Err(result),
            _ => Ok(result),
        }
    }
}

#[repr(C, align(64))]
struct Aligned([u64; 8]);

const CXL_PCIE_BAR_PATH: &CStr = c"/sys/devices/pci0000:27/0000:27:00.1/resource2";
const PAGE_SIZE: usize = 1 << 12;

#[derive(Debug)]
pub struct Csr {
    address_virt: *mut u64,
}

impl Csr {
    const RD_BUFF: usize = 13;
    const WR_BUFF: usize = 14;

    pub fn new() -> io::Result<Self> {
        unsafe {
            let fd = match libc::open(CXL_PCIE_BAR_PATH.as_ptr(), libc::O_RDWR | libc::O_SYNC) {
                -1 => return Err(io::Error::last_os_error()),
                fd => OwnedFd::from_raw_fd(fd),
            };

            let address_virt = match libc::mmap(
                ptr::null_mut(),
                1 << 21,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            ) {
                libc::MAP_FAILED => return Err(io::Error::last_os_error()),
                address => address.cast(),
            };

            Ok(Self { address_virt })
        }
    }

    pub fn set(&mut self, index: usize, value: u64) {
        unsafe { self.address_virt.add(index).write_volatile(value) }
    }
}

#[derive(Debug)]
pub struct Mcas {
    read: Buffer,
    write: Buffer,
}

unsafe impl Sync for Mcas {}
unsafe impl Send for Mcas {}

impl Mcas {
    pub fn new(csr: &mut Csr) -> io::Result<Self> {
        Ok(Self {
            read: Buffer::read(csr)?,
            write: Buffer::write(csr)?,
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Buffer {
    address_phys: *mut libc::c_void,
    address_virt: *mut libc::c_void,
}

unsafe impl Sync for Buffer {}
unsafe impl Send for Buffer {}

impl Buffer {
    pub fn read(csr: &mut Csr) -> io::Result<Self> {
        Self::map(
            csr,
            Some(Csr::RD_BUFF),
            c"/proc/mcas_rd_buff",
            PAGE_SIZE * 16,
        )
    }

    pub fn write(csr: &mut Csr) -> io::Result<Self> {
        Self::map(
            csr,
            Some(Csr::WR_BUFF),
            c"/proc/mcas_wr_buff",
            PAGE_SIZE * 16,
        )
    }

    pub fn target(csr: &mut Csr) -> io::Result<Self> {
        let buffer = Self::map(csr, None, c"/proc/mcas_target_buff", PAGE_SIZE * 16)?;

        unsafe {
            libc::memset(buffer.address_virt.cast(), 0, PAGE_SIZE * 16);
        }

        Ok(buffer)
    }

    fn translate(&self, address: *mut u64) -> u64 {
        (address as u64)
            .checked_sub(self.address_virt as u64)
            .unwrap()
            + self.address_phys as u64
    }

    fn map(csr: &mut Csr, index: Option<usize>, name: &CStr, size: usize) -> io::Result<Self> {
        unsafe {
            let fd = match libc::open(name.as_ptr(), libc::O_RDWR) {
                -1 => return Err(io::Error::last_os_error()),
                fd => OwnedFd::from_raw_fd(fd),
            };

            let mut address_phys = [0u8; 8];
            assert_eq!(
                libc::read(
                    fd.as_raw_fd(),
                    &mut address_phys as *mut u8 as *mut ffi::c_void,
                    8
                ),
                8
            );
            let address_phys = u64::from_ne_bytes(address_phys);

            if let Some(index) = index {
                csr.set(index, address_phys);
            }

            let address_virt = match libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            ) {
                libc::MAP_FAILED => return Err(io::Error::last_os_error()),
                address => address.cast(),
            };

            Ok(Self {
                address_phys: address_phys as *mut _,
                address_virt,
            })
        }
    }
}
