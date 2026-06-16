

// The 'pub' keyword makes this accessible to the CLI and future iOS apps.
pub mod generation;
pub use generation::{GenerationEngine, Gpt2BlockWeights};
// Add this line at the top with your other mod declarations
pub mod profiler;
pub mod kv_cache; // Import kv_cache module
pub mod backend; 
// Re-export the macros for easy use
pub use profiler::print_profile_summary;

pub mod model;
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

// ============================================================
// MEMORY ARENA / WORKSPACE STRUCTURES
// ============================================================

/// Pre-allocated scratch buffers used during a forward pass to eliminate allocations.
pub struct ComputeWorkspace {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub attn_scores: Tensor,
    pub attn_output: Tensor,
    pub ffn_hidden: Tensor,
    pub ffn_output: Tensor,
}

impl ComputeWorkspace {
    /// Allocates the maximum possible memory the model will ever need for one layer.
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let max_elements = max_seq_len * d_model;
        let max_attn_elements = max_seq_len * max_seq_len;
        let max_ffn_elements = max_seq_len * (d_model * 4); // GPT-2 FFN expands by 4x

        Self {
            q_proj: Tensor::new(vec![0.0; max_elements], vec![max_seq_len, d_model]),
            k_proj: Tensor::new(vec![0.0; max_elements], vec![max_seq_len, d_model]),
            v_proj: Tensor::new(vec![0.0; max_elements], vec![max_seq_len, d_model]),
            
            attn_scores: Tensor::new(vec![0.0; max_attn_elements], vec![max_seq_len, max_seq_len]),
            attn_output: Tensor::new(vec![0.0; max_elements], vec![max_seq_len, d_model]),
            
            ffn_hidden: Tensor::new(vec![0.0; max_ffn_elements], vec![max_seq_len, d_model * 4]),
            ffn_output: Tensor::new(vec![0.0; max_elements], vec![max_seq_len, d_model]),
        }
    }

    /// Dynamically resizes the "views" of all buffers for the current sequence length.
    pub fn resize_for_step(&mut self, current_seq_len: usize, d_model: usize) {
        self.q_proj.set_view(vec![current_seq_len, d_model]);
        self.k_proj.set_view(vec![current_seq_len, d_model]);
        self.v_proj.set_view(vec![current_seq_len, d_model]);
        
        self.attn_scores.set_view(vec![current_seq_len, current_seq_len]);
        self.attn_output.set_view(vec![current_seq_len, d_model]);
        
        self.ffn_hidden.set_view(vec![current_seq_len, d_model * 4]);
        self.ffn_output.set_view(vec![current_seq_len, d_model]);
    }
}

impl Tensor {
    // A constructor function to easily create new Tensors
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    // This method takes a row and col, and returns the float at that position
    pub fn get(&self, row: usize, col: usize) -> f32 {
        let cols = self.shape[1]; // Get the 'C' from our shape
        let flat_index = row * cols + col; // Our math formula!
        self.data[flat_index] // Return the data at that index
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
    
    // add this for profiling matmul
    let _guard = crate::profiler::ProfileGuard::new("matmul");

    assert_eq!(self.shape[1], other.shape[0], "Matrix dimensions do not match");
    
    // initially using static block for tiled Matmul 
    let block = 32;
    let out_rows = self.shape[0];
    let out_cols = other.shape[1];
    let shared_dim = self.shape[1];

    // Pre-allocate the output vector
    let mut out_data: Vec<f32> = vec![0.0; out_rows * out_cols];
    // Track this allocation
    crate::profiler::track_allocation(out_data.len() * std::mem::size_of::<f32>());
    // 1. FIXED: Removed explicit type annotations from the loop variables
    for i_block in (0..out_rows).step_by(block){
        for j_block in (0..out_cols).step_by(block){
            for k_block in (0..shared_dim).step_by(block){
                let i_end = (i_block+ block).min(out_rows);
                let j_end = (j_block + block).min(out_cols);
                let k_end = (k_block + block).min(shared_dim);

                for i in i_block..i_end{
                    for j in j_block..j_end{
                        let mut sum = 0.0;
                        for k in k_block..k_end{
                            sum += self.get(i,k) * other.get(k,j);
                        }
                        let flat_index = i * out_cols + j;
                        out_data[flat_index] += sum;
                    }
                }
            }
        }
    }

    // 2. FIXED: Construct and return the new Tensor without a semicolon
    Tensor::new(out_data, vec![out_rows, out_cols])
}

    pub fn add(&self, othervec: &Tensor) -> Tensor{
        // 1. Strict shape matching (we will handle broadcasting later)
        assert_eq!(self.shape, othervec.shape, "Shapes must match for basic addition");
        // 2. Idiomatic Rust element-wise addition
        let out_data: Vec<f32> = self.data
        .iter() // create an iterator onn first vector 
        .zip(othervec.data.iter()) // Pair it up with the second vector's iterator
        .map(|(a, b)| a + b) // Add the paired elements together
        .collect(); // Gather the results back into a new Vec<f32>

        // 3. Return the new Tensor. 
        // We use .clone() on the shape because the new tensor shares the same dimensions.
        Tensor::new(out_data, self.shape.clone())
    }

    pub fn relu(&self) -> Tensor{
        let out_data: Vec<f32> = self.data
        .iter()
        .map(|&x| x.max(0.0))
        .collect();

        Tensor::new(out_data, self.shape.clone())
    }

    pub fn softmax(&self) -> Tensor {
        // Add this for Prifiling Softmax
        let _guard = crate::profiler::ProfileGuard::new("softmax");
        // 1 Gaurd : This implimentation expectes a 2D tensor [row, cols]
        assert_eq!(self.shape.len(), 2, "softmax currently support only 2D tensors");

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut out_data = vec![0.0_f32;rows*cols];

        // Track allocation
        crate::profiler::track_allocation(self.data.len() * std::mem::size_of::<f32>());
        
        // 2. Process each row independently
        
        for i in 0..rows{
            let row_start = i * cols;
            let row_end = row_start + cols;
            let row = &self.data[row_start..row_end];

            // Section A Find MAX

            let max_val = row.iter().fold(f32::NEG_INFINITY, |acc, &x| if x > acc { x } else { acc });

            // section B Exponential
            let exps: Vec<f32> = row
                .iter()
                .map(|&x| (x-max_val).exp())
                .collect();

            // Section C Sum Exponentials
            let sum: f32 = exps.iter().sum();

            // Section D Normalize
            if sum == 0.0 {
                for j in 0..cols{
                    out_data[row_start + j] = 0.0;
                }
            }else {
                    for j in 0..cols{
                        out_data[row_start + j] = exps[j]/sum;
                    }
                }
        
        }
        Tensor::new(out_data, self.shape.clone())
    }

    pub fn transpose(&self) -> Tensor {
    // Guard: this implementation expects a 2D tensor
    assert_eq!(self.shape.len(), 2, "Transpose currently only supports 2D tensors");
    
    let rows = self.shape[0];
    let cols = self.shape[1];
    
    // ──── WHY WE ALLOCATE NEW MEMORY ────
    // Transposing changes the memory layout. In row-major order:
    //   Original: A[i][j] = data[i * cols + j]
    //   Transposed: A^T[i][j] = A[j][i] = data[j * cols + i]
    //
    // We CANNOT just reuse the same Vec<f32> because the element ordering changes.
    // (Future optimization: return a "view" with strides instead of copying.
    //  This is what MLX and PyTorch do. But for now, copy.)
    
    let mut out_data = vec![0.0_f32; rows * cols];
    
    // ──── THE INDEX MAPPING ────
    // For each position (i, j) in the OUTPUT (transposed) matrix:
    //   output position: i * rows + j   (because output has [cols, rows] shape)
    //   corresponding input element: A[j][i] = j * cols + i
    //
    // Wait—let me re-explain with clear variable names.
    // Output shape is [cols, rows] (swapped).
    // For each position (new_row, new_col) in output:
    //   new_row ranges 0..cols (the old column index)
    //   new_col ranges 0..rows (the old row index)
    //   output[new_row * rows + new_col] = input[new_col * cols + new_row]
    
    for old_row in 0..rows {
        for old_col in 0..cols {
            // In the output, old_col becomes the row, old_row becomes the column
            let new_row = old_col;
            let new_col = old_row;
            let out_index = new_row * rows + new_col;
            let in_index = old_row * cols + old_col;
            out_data[out_index] = self.data[in_index];
        }
    }
    // New shape: swapped dimensions
    Tensor::new(out_data, vec![cols, rows])
}

    pub fn scale(&self, scalar: f32) -> Tensor {
    let out_data: Vec<f32> = self.data
        .iter()
        .map(|&x| x * scalar)
        .collect();
    
    Tensor::new(out_data, self.shape.clone())
}

    pub fn scaled_dot_product_attention(query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
    // query:  [q_len, head_dim]
    // key:    [kv_len, head_dim]
    // value:  [kv_len, head_dim]
    
    let d_k = query.shape[1] as f32;
    let scale = 1.0_f32 / d_k.sqrt();
    
    // 1. Compute attention scores: Q @ K^T -> [q_len, kv_len]
    let scores = query.matmul(&key.transpose());  
    
    // 2. Scale scores
    let scaled_scores = scores.scale(scale);
    
    let q_len = scaled_scores.shape[0];
    let kv_len = scaled_scores.shape[1];
    let mut masked_data = scaled_scores.data.clone();

    // 3. Apply Causal Mask safely for non-square matrices
    // We compute the absolute global position offset to align the query with historical keys
    let native_offset = kv_len.saturating_sub(q_len);

    for i in 0..q_len {
        for j in 0..kv_len {
            // Causal rule: a query token at position (native_offset + i) 
            // cannot look at a key token at index j if j > (native_offset + i)
            if j > (native_offset + i) {
                masked_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let masked_scores = Tensor::new(masked_data, vec![q_len, kv_len]);
    
    // 4. Softmax row-wise
    let attention_weights = masked_scores.softmax();  // [q_len, kv_len]
    
    // 5. Weighted sum of values: [q_len, kv_len] @ [kv_len, head_dim] -> [q_len, head_dim]
    attention_weights.matmul(value)  
}

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
    let total: usize = new_shape.iter().product();
    assert_eq!(
        total,
        self.data.len(),
        "Reshape size mismatch: {:?} has {} elements, new shape {:?} needs {}",
        self.shape, self.data.len(), new_shape, total
    );
    Tensor::new(self.data.clone(), new_shape)
}

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, epsilon: f32) -> Tensor {
    // This implementation normalizes the LAST dimension of a 2D tensor
    // Input: [rows, dim], gamma: [dim], beta: [dim]
    assert_eq!(self.shape.len(), 2, "LayerNorm expects 2D input");
    assert_eq!(gamma.shape.len(), 1, "Gamma must be 1D");
    assert_eq!(beta.shape.len(), 1, "Beta must be 1D");
    
    // Add this to profile layer_norm
    let _guard = crate::profiler::ProfileGuard::new("layer_norm");

    let rows = self.shape[0];
    let dim = self.shape[1];
    let mut out_data = vec![0.0_f32; rows * dim];
    
    for i in 0..rows {
        let row_start = i * dim;
        let row = &self.data[row_start..row_start + dim];
        
        // 1. Compute mean
        let mean: f32 = row.iter().sum::<f32>() / dim as f32;
        
        // 2. Compute variance
        let var: f32 = row.iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f32>() / dim as f32;
        
        // 3. Normalize
        let std_dev = (var + epsilon).sqrt();
        
        for j in 0..dim {
            let normalized = (row[j] - mean) / std_dev;
            // 4. Scale and shift
            out_data[row_start + j] = gamma.data[j] * normalized + beta.data[j];
        }
    }
    
    Tensor::new(out_data, self.shape.clone())
}

 /// Add a 1D bias to each row of a 2D matrix (broadcast along rows)
    pub fn add_bias(&self, bias: &Tensor) -> Tensor {
    assert_eq!(self.shape.len(), 2, "add_bias expects 2D input");
    assert_eq!(bias.shape.len(), 1, "add_bias expects 1D bias");
    assert_eq!(self.shape[1], bias.shape[0], "Bias dimension must match last dim");
    
    let rows = self.shape[0];
    let cols = self.shape[1];
    let mut out_data = self.data.clone();
    
    for i in 0..rows {
        for j in 0..cols {
            out_data[i * cols + j] += bias.data[j];
        }
    }
    
    Tensor::new(out_data, self.shape.clone())
}


pub fn gelu(&self) -> Tensor {
    let out_data: Vec<f32> = self.data
        .iter()
        .map(|&x| {
            let x3 = x * x * x;
            let c = (2.0_f32 / std::f32::consts::PI).sqrt();
            let inner = c * (x + 0.044715 * x3);
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect();
    Tensor::new(out_data, self.shape.clone())
}

    // ============================================================
    // ZERO-ALLOCATION OPERATIONS (ARENA PATTERN)
    // ============================================================

    /// Zero-allocation addition. Writes result directly into the `out` tensor.
  pub fn add_into(&self, other: &Tensor, out: &mut Tensor) {
        assert_eq!(self.shape, other.shape, "Shapes must match for addition");
        assert_eq!(self.shape, out.shape, "Output shape must match input shape");

        // Calculate the actual number of active elements based on the current shape, 
        // NOT the underlying vector's total maximum capacity.
        let active_elements: usize = self.shape.iter().product();

        for i in 0..active_elements {
            out.data[i] = self.data[i] + other.data[i];
        }
    }

    /// Zero-allocation matrix multiplication. Writes result directly into the `out` tensor.
/// Zero-allocation matrix multiplication. Writes result directly into the `out` tensor.
/// Routes through the compute backend (Scalar/NEON/Metal).
pub fn matmul_into(&self, other: &Tensor, out: &mut Tensor) {
    let _guard = crate::profiler::ProfileGuard::new("matmul_into");

    assert_eq!(self.shape[1], other.shape[0], "Matrix dimensions do not match");

    let out_rows = self.shape[0];
    let out_cols = other.shape[1];
    let shared_dim = self.shape[1];

    assert_eq!(
        out.shape,
        vec![out_rows, out_cols],
        "Output tensor shape mismatch"
    );

    out.data.fill(0.0);

    let block = 32;

    for i_block in (0..out_rows).step_by(block) {
        for j_block in (0..out_cols).step_by(block) {
            for k_block in (0..shared_dim).step_by(block) {
                let i_end = (i_block + block).min(out_rows);
                let j_end = (j_block + block).min(out_cols);
                let k_end = (k_block + block).min(shared_dim);

                for i in i_block..i_end {
                    for j in j_block..j_end {
                        let mut k = k_block;
                        let mut sum;

                        // ── NEON SIMD path ──
                        #[cfg(target_arch = "aarch64")]
                        {
                            sum = neon_dot(self, other, i, j, k_block, k_end);
                            k = k_end; // NEON handled all of k including cleanup
                        }

                        #[cfg(not(target_arch = "aarch64"))]
                        {
                            sum = 0.0;
                        }

                        // scalar cleanup for non-aarch64
                        while k < k_end {
                            sum += self.get(i, k) * other.get(k, j);
                            k += 1;
                        }

                        out.data[i * out_cols + j] += sum;
                    }
                }
            }
        }
    }
}

    /// Adjusts the active "view" of the tensor without reallocating memory.
pub fn set_view(&mut self, new_shape: Vec<usize>) {
        let required_elements: usize = new_shape.iter().product();
        assert!(
            required_elements <= self.data.len(),
            "Cannot set view: required {} elements, but buffer only holds {}",
            required_elements, self.data.len()
        );
        self.shape = new_shape;
    }

    /// Zero-allocation, in-place bias addition. Mutates the tensor directly.
    pub fn add_bias_in_place(&mut self, bias: &Tensor) {
        assert_eq!(self.shape.len(), 2, "add_bias_in_place expects 2D input");
        assert_eq!(bias.shape.len(), 1, "add_bias_in_place expects 1D bias");
        assert_eq!(self.shape[1], bias.shape[0], "Bias dimension must match last dim");
        
        let rows = self.shape[0];
        let cols = self.shape[1];
        
        for i in 0..rows {
            for j in 0..cols {
                self.data[i * cols + j] += bias.data[j];
            }
        }
    }

}
/// NEON-accelerated dot product for one (i, j) output element.
/// Processes k in chunks of 4 using vfmaq_f32, scalar cleanup for remainder.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon_dot_inner(
    a: &Tensor,
    b: &Tensor,
    i: usize,
    j: usize,
    k_start: usize,
    k_end: usize,
) -> f32 {
    use std::arch::aarch64::*;

    unsafe{
    let mut sum_vec = vdupq_n_f32(0.0);
    let mut k = k_start;

    while k + 4 <= k_end {
        let a0 = *a.data.get_unchecked(i * a.shape[1] + k);
        let a1 = *a.data.get_unchecked(i * a.shape[1] + k + 1);
        let a2 = *a.data.get_unchecked(i * a.shape[1] + k + 2);
        let a3 = *a.data.get_unchecked(i * a.shape[1] + k + 3);
        let a_vec = vld1q_f32([a0, a1, a2, a3].as_ptr());

        let b0 = *b.data.get_unchecked(k       * b.shape[1] + j);
        let b1 = *b.data.get_unchecked((k + 1) * b.shape[1] + j);
        let b2 = *b.data.get_unchecked((k + 2) * b.shape[1] + j);
        let b3 = *b.data.get_unchecked((k + 3) * b.shape[1] + j);
        let b_vec = vld1q_f32([b0, b1, b2, b3].as_ptr());

        sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
        k += 4;
    }
    

    // Horizontal reduction
    let mut sum = vaddvq_f32(sum_vec);

    // Scalar cleanup for remainder
    while k < k_end {
        sum += a.data[i * a.shape[1] + k] * b.data[k * b.shape[1] + j];
        k += 1;
    }

    sum
}
}

/// Safe wrapper — callable from safe Rust inside matmul_into
#[cfg(target_arch = "aarch64")]
fn neon_dot(
    a: &Tensor,
    b: &Tensor,
    i: usize,
    j: usize,
    k_start: usize,
    k_end: usize,
) -> f32 {
    unsafe { neon_dot_inner(a, b, i, j, k_start, k_end) }
}

/// Multi-Head Self-Attention
/// 
/// Shapes:
///   x:           [seq_len, d_model]
///   q_weight:    [d_model, d_model]
///   q_bias:      [d_model]
///   k_weight:    [d_model, d_model]
///   k_bias:      [d_model]
///   v_weight:    [d_model, d_model]
///   v_bias:      [d_model]
///   out_weight:  [d_model, d_model]
///   out_bias:    [d_model]
///   num_heads:   number of attention heads
/// Upgraded Multi-Head Self-Attention with KV Caching
/// Upgraded Multi-Head Self-Attention with KV Caching
pub fn multi_head_attention(
    x: &Tensor,
    q_weight: &Tensor, q_bias: &Tensor,
    k_weight: &Tensor, k_bias: &Tensor,
    v_weight: &Tensor, v_bias: &Tensor,
    out_weight: &Tensor, out_bias: &Tensor,
    num_heads: usize,
    kv_cache: &mut crate::kv_cache::LayerKVCache, 
    workspace: &mut ComputeWorkspace,  
    backend: &dyn crate::backend::ComputeBackend, // 👈 THE NEW CABLE
) -> Tensor {
    let seq_len = x.shape[0];    
    let d_model = x.shape[1];
    let head_dim = d_model / num_heads;
    
    assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
    
    // ── ZERO-ALLOCATION Q/K/V PROJECTIONS VIA BACKEND ──
    workspace.q_proj.set_view(vec![seq_len, d_model]);
    workspace.k_proj.set_view(vec![seq_len, d_model]);
    workspace.v_proj.set_view(vec![seq_len, d_model]);

    backend.matmul_into(x, q_weight, &mut workspace.q_proj);
    workspace.q_proj.add_bias_in_place(q_bias);

    backend.matmul_into(x, k_weight, &mut workspace.k_proj);
    workspace.k_proj.add_bias_in_place(k_bias);

    backend.matmul_into(x, v_weight, &mut workspace.v_proj);
    workspace.v_proj.add_bias_in_place(v_bias);
    
    // ==========================================
    // PHASE 1: WRITE TO CACHE
    // ==========================================
    for s in 0..seq_len {
        for h in 0..num_heads {
            let src_offset = s * d_model + h * head_dim;
            let current_k_vector = &workspace.k_proj.data[src_offset..src_offset + head_dim];
            let current_v_vector = &workspace.v_proj.data[src_offset..src_offset + head_dim];
            
            kv_cache.push_k(h, current_k_vector);
            kv_cache.push_v(h, current_v_vector);
        }
        kv_cache.increment_seq_len();
    }

    // ==========================================
    // PHASE 2: COMPUTE ATTENTION
    // ==========================================
    let mut head_outputs: Vec<Tensor> = Vec::new(); 
    let total_seq_len = kv_cache.seq_len(); 
    
    for h in 0..num_heads {
        let full_k_slice = kv_cache.k_slice(h);
        let full_v_slice = kv_cache.v_slice(h);
        
        let k_historical = Tensor::new(full_k_slice.to_vec(), vec![total_seq_len, head_dim]);
        let v_historical = Tensor::new(full_v_slice.to_vec(), vec![total_seq_len, head_dim]);

        let mut q_head_data = vec![0.0; seq_len * head_dim];
        for s in 0..seq_len {
            for d in 0..head_dim {
                q_head_data[s * head_dim + d] = workspace.q_proj.data[s * d_model + h * head_dim + d];
            }
        }
        let q_head = Tensor::new(q_head_data, vec![seq_len, head_dim]);
        
        let attn_out = Tensor::scaled_dot_product_attention(&q_head, &k_historical, &v_historical);
        head_outputs.push(attn_out);
    }

    // ==========================================
    // PHASE 3: CONCATENATE AND PROJECT
    // ==========================================
    let mut concat_data = vec![0.0; seq_len * d_model];
    for s in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..head_dim {
                let src_idx = s * head_dim + d;
                let dst_idx = s * d_model + h * head_dim + d;
                concat_data[dst_idx] = head_outputs[h].data[src_idx]; 
            }
        }
    }
    
    let concat = Tensor::new(concat_data, vec![seq_len, d_model]);
    concat.matmul(out_weight).add_bias(out_bias)
}

/// A single Transformer Block
pub fn transformer_block(
    x: &Tensor,
    // Attention weights
    q_weight: &Tensor, q_bias: &Tensor,
    k_weight: &Tensor, k_bias: &Tensor,
    v_weight: &Tensor, v_bias: &Tensor,
    out_weight: &Tensor, out_bias: &Tensor,
    num_heads: usize,
    // LayerNorm 1
    ln1_gamma: &Tensor, ln1_beta: &Tensor,
    // FFN
    ffn_w1: &Tensor, ffn_b1: &Tensor,  // First linear layer (expansion)
    ffn_w2: &Tensor, ffn_b2: &Tensor,  // Second linear layer (projection)
    // LayerNorm 2
    ln2_gamma: &Tensor, ln2_beta: &Tensor,
    // 👈 ADDED: Pass the specific layer's cache down to attention
    kv_cache: &mut crate::kv_cache::LayerKVCache, 
    workspace: &mut ComputeWorkspace,  // 👈 ADDED pre defined Compuet Workspace
    backend: &dyn crate::backend::ComputeBackend,
) -> Tensor {
    let seq_len = x.shape[0];

    // Self-Attention with residual
    let normed = x.layer_norm(ln1_gamma, ln1_beta, 1e-5);
    let attn_out = multi_head_attention(
        &normed,
        q_weight, q_bias,
        k_weight, k_bias,
        v_weight, v_bias,
        out_weight, out_bias,
        num_heads,
        kv_cache, // 👈 Hand it to the attention function
        workspace, // Pre defined Compter workspace tensor
        backend,
    );
    let residual1 = x.add(&attn_out);  // Skip connection
    


    // Feed-Forward Network with residual
    let normed2 = residual1.layer_norm(ln2_gamma, ln2_beta, 1e-5);
    // 👈 3. UPGRADED FFN TO ZERO-ALLOCATION BACKEND MATH
    workspace.ffn_hidden.set_view(vec![seq_len, ffn_w1.shape[1]]);
    backend.matmul_into(&normed2, ffn_w1, &mut workspace.ffn_hidden);
    workspace.ffn_hidden.add_bias_in_place(ffn_b1);
    
    let ffn_activated = workspace.ffn_hidden.gelu(); 

    workspace.ffn_output.set_view(vec![seq_len, ffn_w2.shape[1]]);
    backend.matmul_into(&ffn_activated, ffn_w2, &mut workspace.ffn_output);
    workspace.ffn_output.add_bias_in_place(ffn_b2);

    let residual2 = residual1.add(&workspace.ffn_output);  // Skip connection
    residual2
}

/// Run GPT-2 forward pass
/// Returns logits: [seq_len, vocab_size]
pub fn gpt2_forward(
    token_ids: &[usize],
    wte: &Tensor,
    wpe: &Tensor,
    blocks: &[(
        Tensor, Tensor,  // 0,1: q_weight, q_bias
        Tensor, Tensor,  // 2,3: k_weight, k_bias
        Tensor, Tensor,  // 4,5: v_weight, v_bias
        Tensor, Tensor,  // 6,7: out_weight, out_bias
        Tensor, Tensor,  // 8,9: ln1_gamma, ln1_beta
        Tensor, Tensor,  // 10,11: ffn_w1, ffn_b1
        Tensor, Tensor,  // 12,13: ffn_w2, ffn_b2
        Tensor, Tensor,  // 14,15: ln2_gamma, ln2_beta
    )],
    ln_f_gamma: &Tensor,
    ln_f_beta: &Tensor,
    lm_head_weight: &Tensor,
    kv_cache: &mut crate::kv_cache::KVCache, // 👈 ADDED: The master cache for all 12 layers
    workspace: &mut ComputeWorkspace,  // 👈 ADDED Pre defined Compute Workspace
    backend: &dyn crate::backend::ComputeBackend // 👈🏻 Added to pass the backend
) -> Tensor {
    // Add this for profiling GPT2 forward pass
    let _total_guard = crate::profiler::ProfileGuard::new("gpt2_forward_total");

    let tok_emb = embedding_lookup(wte, token_ids);
    // 👈 CRITICAL FIX: Positional Embeddings
    // If the cache has 5 tokens, and we feed 1 new token, its position index should be 5, not 0!
    let start_pos = kv_cache.layers[0].seq_len();
    let positions: Vec<usize> = (start_pos..start_pos + token_ids.len()).collect();
    let pos_emb = embedding_lookup(wpe, &positions);
    let mut hidden = tok_emb.add(&pos_emb);
    
    for (i,block) in blocks.iter().enumerate() {
        hidden = transformer_block(
            &hidden,
            &block.0, &block.1,   // q_w, q_b ✅
            &block.2, &block.3,   // k_w, k_b ✅
            &block.4, &block.5,   // v_w, v_b ✅
            &block.6, &block.7,   // out_w, out_b ✅
            12,                    // n_head
            &block.8, &block.9,   // ln1_g, ln1_b ✅
            &block.10, &block.11, // ffn_w1, ffn_b1 ✅
            &block.12, &block.13, // ffn_w2, ffn_b2 ✅
            &block.14, &block.15, // ln2_g, ln2_b ✅
            &mut kv_cache.layers[i], // 👈 Pass the SPECIFIC cache for this layer
            workspace,  // 👈 ADDEd Pre defined Compute Workspace
            backend, // 👈🏻 Added to pass backend
        );
    }
    
    let normed = hidden.layer_norm(ln_f_gamma, ln_f_beta, 1e-5);

    // Profiling GPT2 Forward Pass
    // crate::profiler::print_profile_summary();

    normed.matmul(&lm_head_weight.transpose())
}
pub fn embedding_lookup(embedding_table: &Tensor, token_ids: &[usize]) -> Tensor {
    let d_model = embedding_table.shape[1];
    let seq_len = token_ids.len();
    let mut out_data = vec![0.0_f32; seq_len * d_model];
    
    for (i, &token_id) in token_ids.iter().enumerate() {
        if token_id >= embedding_table.shape[0] {
            panic!("Token ID {} out of range (vocab size: {})", token_id, embedding_table.shape[0]);
        }
        let src_start = token_id * d_model;
        let dst_start = i * d_model;
        out_data[dst_start..dst_start + d_model]
            .copy_from_slice(&embedding_table.data[src_start..src_start + d_model]);
    }
    
    Tensor::new(out_data, vec![seq_len, d_model])
}

// ============================================================
// UTILITY: Split GPT-2 fused QKV weights
// ============================================================

/// Split fused QKV weight matrix into separate Q, K, V matrices.
/// GPT-2 stores Q, K, V as one big matrix:
///   c_attn.weight: [d_model, 3 * d_model]
///   c_attn.bias:   [3 * d_model]
/// We split them into three separate [d_model, d_model] weights and [d_model] biases.
pub fn split_qkv(
    fused_weight: &Tensor,
    fused_bias: &Tensor,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) {
    let _d_model = fused_weight.shape[0];
    let total_dim = fused_weight.shape[1];
    let third = total_dim / 3;

    assert_eq!(total_dim % 3, 0, "Fused QKV dimension must be divisible by 3");

    // Split weight columns: [d_model, 3*d_model] -> three [d_model, d_model]
    let q_w = extract_columns(fused_weight, 0, third);
    let k_w = extract_columns(fused_weight, third, third * 2);
    let v_w = extract_columns(fused_weight, third * 2, third * 3);

    // Split bias: [3*d_model] -> three [d_model]
    let q_b = Tensor::new(fused_bias.data[0..third].to_vec(), vec![third]);
    let k_b = Tensor::new(fused_bias.data[third..third * 2].to_vec(), vec![third]);
    let v_b = Tensor::new(fused_bias.data[third * 2..third * 3].to_vec(), vec![third]);

    (q_w, q_b, k_w, k_b, v_w, v_b)
}

/// Extract a range of columns from a 2D matrix.
/// matrix: [rows, src_cols] -> returns: [rows, col_end - col_start]
fn extract_columns(matrix: &Tensor, col_start: usize, col_end: usize) -> Tensor {
    let rows = matrix.shape[0];
    let src_cols = matrix.shape[1];
    let out_cols = col_end - col_start;
    let mut out_data = vec![0.0_f32; rows * out_cols];

    for r in 0..rows {
        let src_start = r * src_cols + col_start;
        let dst_start = r * out_cols;
        out_data[dst_start..dst_start + out_cols]
            .copy_from_slice(&matrix.data[src_start..src_start + out_cols]);
    }

    Tensor::new(out_data, vec![rows, out_cols])
}