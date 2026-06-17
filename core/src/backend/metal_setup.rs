// This is. a Rust to GPU dispatcher for inference Engine
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ffi::c_void;
use std::mem;

pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline_state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MetalContext {
    pub fn new() -> Self {
        // 1. Grab the M4 GPU hardware
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        let command_queue = device.newCommandQueue().unwrap();

        // 2. Load the shader source code
        let source = include_str!("../matmul.metal");
        let ns_source = NSString::from_str(source);
        let compile_options = MTLCompileOptions::new();
        
        // 3. Compile the MSL code into an executable library
        let library = unsafe {
            device.newLibraryWithSource_options_error(&ns_source, Some(&compile_options)).unwrap()
        };

        let function_name = NSString::from_str("naive_matmul");
        let function = library.newFunctionWithName(&function_name).unwrap();

        let pipeline_state = unsafe {
            device.newComputePipelineStateWithFunction_error(&function).unwrap()
        };

        Self { device, command_queue, pipeline_state }
    }

    pub fn dispatch_matmul(&self, a_data: &[f32], b_data: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let a_size = a_data.len() * mem::size_of::<f32>();
        let b_size = b_data.len() * mem::size_of::<f32>();
        let out_size = m * n * mem::size_of::<f32>();
        let dim_size = 3 * mem::size_of::<u32>();

        unsafe {
            // 1. Map Rust memory into GPU Unified Memory
            let a_buffer = self.device.newBufferWithBytes_length_options(
                a_data.as_ptr() as *const c_void,
                a_size as _,
                MTLResourceOptions::StorageModeShared, // 👈 UMA Magic
            ).unwrap();

            let b_buffer = self.device.newBufferWithBytes_length_options(
                b_data.as_ptr() as *const c_void,
                b_size as _,
                MTLResourceOptions::StorageModeShared,
            ).unwrap();

            let out_buffer = self.device.newBufferWithLength_options(
                out_size as _,
                MTLResourceOptions::StorageModeShared,
            ).unwrap();

            let dimensions: [u32; 3] = [m as u32, n as u32, k as u32];
            let dim_buffer = self.device.newBufferWithBytes_length_options(
                dimensions.as_ptr() as *const c_void,
                dim_size as _,
                MTLResourceOptions::StorageModeShared,
            ).unwrap();

            // 2. Encode instructions for the GPU
            let command_buffer = self.command_queue.commandBuffer().unwrap();
            let encoder = command_buffer.computeCommandEncoder().unwrap();

            encoder.setComputePipelineState(&self.pipeline_state);
            encoder.setBuffer_offset_atIndex(Some(&a_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&b_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&out_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&dim_buffer), 0, 3);

            // 3. Define the Grid and Fire
            let grid_size = MTLSize { width: n, height: m, depth: 1 };
            let threadgroup_size = MTLSize { width: 16, height: 16, depth: 1 };

            encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
            encoder.endEncoding();
            command_buffer.commit();
            
            // Wait for GPU to finish
            command_buffer.waitUntilCompleted();

            // 4. Read result back to Rust
            let ptr = out_buffer.contents() as *const f32;
            let mut result = vec![0.0; m * n];
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), m * n);
            result
        }
    }
}