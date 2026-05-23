use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use std::cell::RefCell;

// Global counters for aggregate statistics
static MATMUL_TIME: AtomicU64 = AtomicU64::new(0);
static MATMUL_CALLS: AtomicU64 = AtomicU64::new(0);
static SOFTMAX_TIME: AtomicU64 = AtomicU64::new(0);
static SOFTMAX_CALLS: AtomicU64 = AtomicU64::new(0);
static LAYERNORM_TIME: AtomicU64 = AtomicU64::new(0);
static LAYERNORM_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_BYTES: AtomicU64 = AtomicU64::new(0);

// Thread-local for nested timing
thread_local! {
    static TIMER_STACK: RefCell<Vec<(&'static str, Instant)>> = RefCell::new(Vec::new());
}

pub struct ProfileGuard {
    name: &'static str,
    start: Instant,
}

impl ProfileGuard {
    pub fn new(name: &'static str) -> Self {
        let start = Instant::now();
        
        // Push to stack for nesting detection
        TIMER_STACK.with(|stack| {
            stack.borrow_mut().push((name, start));
        });
        
        Self { name, start }
    }
}

impl Drop for ProfileGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;
        
        // Pop from stack
        TIMER_STACK.with(|stack| {
            stack.borrow_mut().pop();
        });
        
        // Update global counters based on function name
        match self.name {
            "matmul" => {
                MATMUL_TIME.fetch_add(elapsed_ns, Ordering::Relaxed);
                MATMUL_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            "softmax" => {
                SOFTMAX_TIME.fetch_add(elapsed_ns, Ordering::Relaxed);
                SOFTMAX_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            "layer_norm" => {
                LAYERNORM_TIME.fetch_add(elapsed_ns, Ordering::Relaxed);
                LAYERNORM_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        
        // Print if it took more than 1ms (helps spot slow operations)
        if elapsed > Duration::from_millis(1) {
            eprintln!("  [FAST] {} took {:.2}ms", self.name, elapsed.as_secs_f64() * 1000.0);
        }
    }
}

// Track memory allocations (call this whenever you create a new Tensor)
pub fn track_allocation(bytes: usize) {
    ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATION_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

// Call this at the end to print summary
pub fn print_profile_summary() {
    eprintln!("\n========== PROFILE SUMMARY ==========");
    
    let matmul_time = MATMUL_TIME.load(Ordering::Relaxed);
    let matmul_calls = MATMUL_CALLS.load(Ordering::Relaxed);
    if matmul_calls > 0 {
        eprintln!("MATMUL: {} calls, total {:.2}ms, avg {:.3}ms per call",
            matmul_calls,
            matmul_time as f64 / 1_000_000.0,
            (matmul_time as f64 / matmul_calls as f64) / 1_000_000.0
        );
    }
    
    let softmax_time = SOFTMAX_TIME.load(Ordering::Relaxed);
    let softmax_calls = SOFTMAX_CALLS.load(Ordering::Relaxed);
    if softmax_calls > 0 {
        eprintln!("SOFTMAX: {} calls, total {:.2}ms, avg {:.3}ms per call",
            softmax_calls,
            softmax_time as f64 / 1_000_000.0,
            (softmax_time as f64 / softmax_calls as f64) / 1_000_000.0
        );
    }
    
    let ln_time = LAYERNORM_TIME.load(Ordering::Relaxed);
    let ln_calls = LAYERNORM_CALLS.load(Ordering::Relaxed);
    if ln_calls > 0 {
        eprintln!("LAYER_NORM: {} calls, total {:.2}ms, avg {:.3}ms per call",
            ln_calls,
            ln_time as f64 / 1_000_000.0,
            (ln_time as f64 / ln_calls as f64) / 1_000_000.0
        );
    }
    
    let alloc_count = ALLOCATION_COUNT.load(Ordering::Relaxed);
    let alloc_bytes = ALLOCATION_BYTES.load(Ordering::Relaxed);
    eprintln!("ALLOCATIONS: {} allocations, {} MB total",
        alloc_count,
        alloc_bytes as f64 / 1_000_000.0
    );
    
    eprintln!("=====================================\n");
}

// Macro for easy usage (put at top of any function you want to profile)
#[macro_export]
macro_rules! profile_fn {
    () => {
        let _guard = $crate::profiler::ProfileGuard::new(function_name!());
    };
}

// Helper macro to get function name as string
#[macro_export]
macro_rules! function_name {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        &name[..name.len() - 3]
    }};
}