use crate::allocator;
use crate::atomic::Convert64;
use crate::atomic::Version;
use crate::cache;
use crate::recover;
use crate::recover::HeapState;
use crate::thread;
use crate::Atomic;

#[repr(C, align(64))]
pub(crate) struct Detectable<T>(Atomic<State<T>>, [u8; 63]);

#[ribbit::pack(size = 64, debug)]
pub(crate) struct State<T> {
    #[ribbit(size = 16)]
    id: Option<thread::Id>,

    #[ribbit(size = 16)]
    version: Version,

    #[ribbit(size = 32)]
    inner: T,
}

impl<T: ribbit::Pack<Loose = L>, L: Convert64> Detectable<T> {
    pub(crate) fn load(&self, context: &allocator::Context) -> T {
        let old = self.0.load();

        cache::flush(&self.0, cache::Invalidate::No);
        cache::fence();

        self.help(context, old);
        old.inner()
    }

    pub(crate) fn store(&self, context: &mut allocator::Context, value: T) {
        let old = self.0.load();
        self.help(context, old);
        self.0
            .store(State::new(Some(context.id), Version::default(), value));

        cache::flush(&self.0, cache::Invalidate::No);
        cache::fence();
    }

    pub(crate) fn update<F, B>(&self, context: &mut allocator::Context, mut next: F) -> Option<T>
    where
        F: FnMut(T, Version) -> Option<(T, HeapState<B>)>,
        recover::State: From<HeapState<B>>,
    {
        let version = context.help.load(context.id, context.id).next();

        if cfg!(feature = "recover-cas") {
            context.help.store(context.id, context.id, version);
        }

        let mut old = self.0.load();
        loop {
            self.help(context, old);

            let (new, log) = next(old.inner(), version)?;

            // Unsync because following compare-exchange is serializing
            context.log_unsync(log);

            match self
                .0
                .compare_exchange(old, State::new(Some(context.id), version, new))
            {
                Err(next) => old = next,
                Ok(_) => {
                    cache::flush(&self.0, cache::Invalidate::No);
                    cache::fence();
                    return Some(old.inner());
                }
            }
        }
    }

    fn help(&self, context: &allocator::Context, state: State<T>) {
        if !cfg!(feature = "recover-cas") {
            return;
        }

        let Some(id) = state.id() else { return };
        let version = state.version();
        context.help.store(context.id, id, version);
    }
}

pub(crate) mod help {
    use core::sync::atomic::AtomicU16;
    use core::sync::atomic::Ordering;

    use crate::atomic::Version;
    use crate::cache;
    use crate::thread;

    pub(crate) struct Array(crate::thread::Array<[AtomicU16; crate::COUNT_THREAD]>);

    impl Array {
        pub(super) fn load(&self, i: thread::Id, j: thread::Id) -> Version {
            let version = self.0[i][u16::from(j) as usize].load(Ordering::Relaxed);
            Version::new(version)
        }

        pub(super) fn store(&self, i: thread::Id, j: thread::Id, new: Version) {
            let version = &self.0[i][u16::from(j) as usize];

            version.store(ribbit::private::pack(new), Ordering::Relaxed);

            cache::flush(version, cache::Invalidate::No);
            cache::fence();

            cache::flush_cxl(version);
            cache::fence_cxl();
        }
    }
}
