// src/backend/mod.rs
pub mod metal_setup;
use crate::Tensor;

pub trait ComputeBackend: Send + Sync {
    fn matmul_into(&self, a: &Tensor, b: &Tensor, out: &mut Tensor);
    fn name(&self) -> &'static str;
}

// ── Scalar fallback (The baseline CPU math) ──
pub struct ScalarBackend;

impl ComputeBackend for ScalarBackend {
    fn name(&self) -> &'static str { "scalar" }
    
    fn matmul_into(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) {
        let _guard = crate::profiler::ProfileGuard::new("matmul_into_scalar");
        let out_rows = a.shape[0];
        let out_cols = b.shape[1];
        let shared_dim = a.shape[1];
        
        out.data.fill(0.0);

        // Cache-friendly i -> k -> j loop
        for i in 0..out_rows {
            for k in 0..shared_dim {
                let a_val = a.data[i * shared_dim + k];
                for j in 0..out_cols {
                    out.data[i * out_cols + j] += a_val * b.data[k * out_cols + j];
                }
            }
        }
    }
}

// ── NEON backend (ARM CPU SIMD Acceleration) ──
pub struct NeonBackend;

impl ComputeBackend for NeonBackend {
    fn name(&self) -> &'static str { "neon" }
    
    fn matmul_into(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) {
        #[cfg(target_arch = "aarch64")]
        {
            let _guard = crate::profiler::ProfileGuard::new("matmul_into_neon");
            let out_rows = a.shape[0];
            let out_cols = b.shape[1];
            let shared_dim = a.shape[1];
            
            out.data.fill(0.0);

            unsafe {
                use std::arch::aarch64::*;
                
                // Cache-friendly i -> k -> j loop
                for i in 0..out_rows {
                    for k in 0..shared_dim {
                        // A is constant across the inner row loop, so we broadcast it
                        // to fill an entire 128-bit register (4 identical floats)
                        let a_val = a.data[i * shared_dim + k];
                        let a_vec = vdupq_n_f32(a_val); 

                        let mut j = 0;
                        
                        // Process chunks of 4 floats simultaneously
                        while j + 4 <= out_cols {
                            // Contiguous memory load for matrix B
                            let b_ptr = b.data.as_ptr().add(k * out_cols + j);
                            let b_vec = vld1q_f32(b_ptr);
                            
                            // Contiguous memory load/store for Output matrix
                            let out_ptr = out.data.as_mut_ptr().add(i * out_cols + j);
                            let mut out_vec = vld1q_f32(out_ptr);
                            
                            // Fused Multiply-Add: out_vec = out_vec + (a_vec * b_vec)
                            out_vec = vfmaq_f32(out_vec, a_vec, b_vec);
                            
                            // Write the 4 calculated floats back to memory
                            vst1q_f32(out_ptr, out_vec);
                            
                            j += 4;
                        }

                        // Scalar cleanup for remainders (e.g., if out_cols is 15, this handles the last 3)
                        while j < out_cols {
                            out.data[i * out_cols + j] += a_val * b.data[k * out_cols + j];
                            j += 1;
                        }
                    }
                }
            }
        }
        
        // If someone compiles this on an Intel Mac or Windows PC, fallback safely
        #[cfg(not(target_arch = "aarch64"))]
        {
            ScalarBackend.matmul_into(a, b, out);
        }
    }
}

// ── Metal backend (stub, wired later) ──
pub struct MetalBackend{
    context: metal_setup::MetalContext,
}

impl ComputeBackend for MetalBackend {
    fn name(&self) -> &'static str { "metal" }
    fn matmul_into(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) {
        NeonBackend.matmul_into(a, b, out)
    }
}

// ── Detection: called once at startup ──
pub fn select_backend() -> Box<dyn ComputeBackend> {
    #[cfg(target_arch = "aarch64")]
    {
        println!("Backend: NEON (aarch64 detected)");
        return Box::new(NeonBackend);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        println!("Backend: scalar (fallback)");
        Box::new(ScalarBackend)
    }
}