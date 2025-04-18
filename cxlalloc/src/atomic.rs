use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ops::DerefMut;
// use core::ops::Shl as _;
// use core::ops::Sub as _;
use core::sync::atomic::AtomicU64;
use core::sync::atomic::Ordering;

#[repr(C)]
#[derive(Debug)]
pub struct Atomic<T> {
    value: AtomicU64,
    _type: PhantomData<T>,
}

impl<T> Atomic<T> {
    pub const unsafe fn new_unchecked(value: u64) -> Self {
        Self {
            value: AtomicU64::new(value),
            _type: PhantomData,
        }
    }
}

impl<T: ribbit::Pack<Loose = L>, L: Convert64> Atomic<T> {
    pub fn new(value: T) -> Self {
        let value = ribbit::private::pack(value).into_u64();
        Self {
            value: AtomicU64::new(value),
            _type: PhantomData,
        }
    }

    pub fn load(&self) -> T {
        unsafe { Self::unpack(self.value.load(Ordering::Acquire)) }
    }

    pub fn store(&self, value: T) {
        self.value.store(Self::pack(value), Ordering::Release)
    }

    pub fn compare_exchange(&self, old: T, new: T) -> Result<T, T> {
        // const TSC_FREQUENCY: u64 = 2_803_200_000;
        // const LATENCY_NS: u64 = 100;
        // const LATENCY_CYCLE: u64 = (LATENCY_NS << 32) / TSC_FREQUENCY;
        //
        // let mut core = 0;
        // let start = unsafe { core::arch::x86_64::__rdtscp(&mut core) };
        // while unsafe { core::arch::x86_64::__rdtscp(&mut core) }
        //     .sub(start)
        //     .shl(32u32)
        //     .ge(&LATENCY_CYCLE)
        // {}

        crate::raw::mcas(self.value.as_ptr(), Self::pack(old), Self::pack(new))
            .map(|value| unsafe { Self::unpack(value) })
            .map_err(|value| unsafe { Self::unpack(value) })

        // self.value
        //     .compare_exchange(
        //         Self::pack(old),
        //         Self::pack(new),
        //         Ordering::AcqRel,
        //         Ordering::Acquire,
        //     )
        //     .map(|value| unsafe { Self::unpack(value) })
        //     .map_err(|value| unsafe { Self::unpack(value) })
    }

    pub fn fetch_xor(&self, value: u64) -> u64 {
        self.value.fetch_xor(value, Ordering::AcqRel)
    }

    fn pack(value: T) -> u64 {
        ribbit::private::pack(value).into_u64()
    }

    unsafe fn unpack(value: u64) -> T {
        ribbit::private::unpack(L::from_u64(value))
    }
}

pub trait Convert64 {
    fn from_u64(value: u64) -> Self;
    fn into_u64(self) -> u64;
}

macro_rules! impl_convert64 {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Convert64 for $ty {
                fn from_u64(value: u64) -> Self {
                    value as $ty
                }

                fn into_u64(self) -> u64 {
                    self as u64
                }
            }
        )*
    };
}

impl_convert64!(u8, u16, u32, u64);

#[repr(C)]
#[ribbit::pack(size = 16, debug, eq, hash)]
#[derive(Default)]
pub struct Version(u16);

impl Version {
    pub fn next(&self) -> Self {
        Self::new(self._0().wrapping_add(1))
    }
}

#[repr(align(64))]
pub struct Pad<T>(T);

impl<T> Deref for Pad<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Pad<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
