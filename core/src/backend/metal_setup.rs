use objc2::rc::Retained;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ffi::c_void;
use std::mem;
use objc2::runtime::ProtocolObject;
use std::ptr::NonNull;

pub struct MetalContext {
    // Retained<T> is objc2's smart pointer. It handles Automatic Reference Counting (ARC) natively.
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline_state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MetalContext {
    pub fn new() -> Self {
        // 1. Grab the Apple Silicon GPU
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device found");
        
        // 2. Create the Command Queue (the mailbox where we drop GPU tasks)
        let command_queue = device.newCommandQueue().unwrap();

        // 3. Load and Compile the MSL code via Just-In-Time (JIT) compilation
        let source = include_str!("../matmul.metal");
        let ns_source = NSString::from_str(source);
        let compile_options = MTLCompileOptions::new();
        
        let library = unsafe {
            device.newLibraryWithSource_options_error(&ns_source, Some(&compile_options)).unwrap()
        };

        // 4. Extract the specific function and bake it into a Pipeline State
        let function_name = NSString::from_str("naive_matmul");
        let function = library.newFunctionWithName(&function_name).unwrap();

        let pipeline_state = unsafe {
            device.newComputePipelineStateWithFunction_error(&function).unwrap()
        };

        Self { device, command_queue, pipeline_state }
    }

    /// Takes raw Rust slices, wraps them in GPU buffers, and fires the shader.
        /// Takes raw Rust slices, wraps them in GPU buffers, and fires the shader.
    pub fn dispatch_matmul(&self, a_data: &[f32], b_data: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let a_size = a_data.len() * mem::size_of::<f32>();
        let b_size = b_data.len() * mem::size_of::<f32>();
        let out_size = m * n * mem::size_of::<f32>();
        let dim_size = 3 * mem::size_of::<u32>();

        // 1. Re-declare the dimensions array (this was missing in your pasted code!)
        let dimensions: [u32; 3] = [m as u32, n as u32, k as u32];

        unsafe {
            // 2. Wrap pointers inline to guarantee the compiler sees NonNull<c_void>
            let a_buffer = self.device.newBufferWithBytes_length_options(
                NonNull::new(a_data.as_ptr() as *mut c_void).unwrap(),
                a_size,
                MTLResourceOptions::StorageModeShared,
            ).unwrap();

            let b_buffer = self.device.newBufferWithBytes_length_options(
                NonNull::new(b_data.as_ptr() as *mut c_void).unwrap(),
                b_size,
                MTLResourceOptions::StorageModeShared,
            ).unwrap();

            let out_buffer = self.device.newBufferWithLength_options(
                out_size,
                MTLResourceOptions::StorageModeShared,
            ).unwrap();

            let dim_buffer = self.device.newBufferWithBytes_length_options(
                NonNull::new(dimensions.as_ptr() as *mut c_void).unwrap(),
                dim_size,
                MTLResourceOptions::StorageModeShared,
            ).unwrap();

            // 3. ENCODE THE COMMANDS
            let command_buffer = self.command_queue.commandBuffer().unwrap();
            let encoder = command_buffer.computeCommandEncoder().unwrap();

            encoder.setComputePipelineState(&self.pipeline_state);
            
            // Map the Rust buffers to the [[buffer(X)]] slots in matmul.metal
            encoder.setBuffer_offset_atIndex(Some(&a_buffer), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&b_buffer), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&out_buffer), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&dim_buffer), 0, 3);

            // 4. CONFIGURE THE GRID GEOMETRY
            let grid_size = MTLSize { width: n, height: m, depth: 1 };
            let threadgroup_size = MTLSize { width: 16, height: 16, depth: 1 };

            // 5. FIRE THE SHADER
            encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
            encoder.endEncoding();
            command_buffer.commit();
            
            // 6. WAIT FOR EXECUTION
            command_buffer.waitUntilCompleted();

            // 7. RETRIEVE THE RESULTS
            // CRITICAL FIX: out_buffer.contents() returns a NonNull, so we MUST call .as_ptr() 
            // before we cast it to a raw *const f32 pointer!
            let ptr = out_buffer.contents().as_ptr() as *const f32;
            let mut result = vec![0.0; m * n];
            
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), m * n);
            
            result
        }
    }
}