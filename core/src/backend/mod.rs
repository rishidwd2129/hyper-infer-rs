// src/backend/mod.rs

pub trait ComputeBackend: Send + Sync {
    fn matmul_into(&self, a: &crate::Tensor, b: &crate::Tensor, out: &mut crate::Tensor);
    fn name(&self) -> &'static str;
}

// ── Scalar fallback (what you have today, just moved) ──
pub struct ScalarBackend;

impl ComputeBackend for ScalarBackend {
    fn name(&self) -> &'static str { "scalar" }
    fn matmul_into(&self, a: &crate::Tensor, b: &crate::Tensor, out: &mut crate::Tensor) {
        // exact copy of your current matmul_into body
    }
}

// ── NEON backend (stub for now, real impl next) ──
pub struct NeonBackend;

impl ComputeBackend for NeonBackend {
    fn name(&self) -> &'static str { "neon" }
    fn matmul_into(&self, a: &crate::Tensor, b: &crate::Tensor, out: &mut crate::Tensor) {
        // TODO: vfmaq_f32 inner loop — falls through to scalar for now
        ScalarBackend.matmul_into(a, b, out)
    }
}

// ── Metal backend (stub, wired later) ──
pub struct MetalBackend;

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &'static str { "metal" }
    fn matmul_into(&self, a: &crate::Tensor, b: &crate::Tensor, out: &mut crate::Tensor) {
        // TODO: MSL shader dispatch
        NeonBackend.matmul_into(a, b, out)
    }
}

// ── Detection: called once at startup ──
pub fn select_backend() -> Box<dyn ComputeBackend> {
    #[cfg(target_arch = "aarch64")]
    {
        // Metal check will go here when we add the metal feature flag
        // #[cfg(feature = "metal")]
        // if MetalBackend::available() { return Box::new(MetalBackend); }
        
        println!("Backend: NEON (aarch64 detected)");
        return Box::new(NeonBackend);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("Backend: scalar (fallback)");
        Box::new(ScalarBackend)
    }
}