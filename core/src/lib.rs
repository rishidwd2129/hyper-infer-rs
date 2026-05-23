// The 'pub' keyword makes this accessible to the CLI and future iOS apps.

// Add this line at the top with your other mod declarations
pub mod profiler;

// Re-export the macros for easy use
pub use profiler::print_profile_summary;

pub mod model;
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
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
    // query:  [seq_len, d_k]   - what we're looking for
    // key:    [seq_len, d_k]   - what we match against
    // value:  [seq_len, d_v]   - what we extract
    
    let d_k = query.shape[1] as f32;
    let scale = 1.0_f32 / d_k.sqrt();
    
    // 1. Compute attention scores: Q @ K^T
    let scores = query.matmul(&key.transpose());  // [seq_len, seq_len]
    
    // 2. Scale scores to keep variance ~1
    let scaled_scores = scores.scale(scale);
    
    // 3. Softmax to get attention weights (row-wise probabilities)
    let attention_weights = scaled_scores.softmax();  // [seq_len, seq_len]
    
    // 4. Weighted sum of values
    attention_weights.matmul(value)  // [seq_len, d_v]
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
pub fn multi_head_attention(
    x: &Tensor,
    q_weight: &Tensor, q_bias: &Tensor,
    k_weight: &Tensor, k_bias: &Tensor,
    v_weight: &Tensor, v_bias: &Tensor,
    out_weight: &Tensor, out_bias: &Tensor,
    num_heads: usize,
) -> Tensor {
    let seq_len = x.shape[0];
    let d_model = x.shape[1];
    let head_dim = d_model / num_heads;
    
    assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
    
    // 1. Linear projections: Q = xW_q + b_q, K = xW_k + b_k, V = xW_v + b_v
    let q = x.matmul(q_weight).add_bias(q_bias);  // [seq_len, d_model]
    let k = x.matmul(k_weight).add_bias(k_bias);
    let v = x.matmul(v_weight).add_bias(v_bias);
    
    // 2. Reshape to separate heads: [seq_len, d_model] → [seq_len, num_heads, head_dim]
    //    Then we treat each head independently
    //    For simplicity, reshape to [num_heads, seq_len, head_dim]
    //    by first going [seq_len, num_heads, head_dim] then "transposing" axes
    
    // Step 2a: Reshape to [seq_len, num_heads, head_dim]
    let _q_3d = q.reshape(vec![seq_len, num_heads, head_dim]);
    let _k_3d = k.reshape(vec![seq_len, num_heads, head_dim]);
    let _v_3d = v.reshape(vec![seq_len, num_heads, head_dim]);
    
    // Step 2b: Manual head extraction using 2D slices
    // Since our ops are 2D, we loop over heads and run attention per head
    
    let mut head_outputs: Vec<Tensor> = Vec::new();
    
    for h in 0..num_heads {
        // Extract head h: Q_h of shape [seq_len, head_dim]
        let mut q_head_data = vec![0.0; seq_len * head_dim];
        let mut k_head_data = vec![0.0; seq_len * head_dim];
        let mut v_head_data = vec![0.0; seq_len * head_dim];
        
        for s in 0..seq_len {
            for d in 0..head_dim {
                let src_idx = s * d_model + h * head_dim + d;
                let dst_idx = s * head_dim + d;
                q_head_data[dst_idx] = q.data[src_idx];
                k_head_data[dst_idx] = k.data[src_idx];
                v_head_data[dst_idx] = v.data[src_idx];
            }
        }
        
        let q_head = Tensor::new(q_head_data, vec![seq_len, head_dim]);
        let k_head = Tensor::new(k_head_data, vec![seq_len, head_dim]);
        let v_head = Tensor::new(v_head_data, vec![seq_len, head_dim]);
        
        // Run attention for this head
        let attn_out = Tensor::scaled_dot_product_attention(&q_head, &k_head, &v_head);
        head_outputs.push(attn_out);
    }
    
    // 3. Concatenate heads back: [num_heads, seq_len, head_dim] → [seq_len, d_model]
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
    
    // 4. Output projection
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
) -> Tensor {
    // Self-Attention with residual
    let normed = x.layer_norm(ln1_gamma, ln1_beta, 1e-5);
    let attn_out = multi_head_attention(
        &normed,
        q_weight, q_bias,
        k_weight, k_bias,
        v_weight, v_bias,
        out_weight, out_bias,
        num_heads,
    );
    let residual1 = x.add(&attn_out);  // Skip connection
    


    // Feed-Forward Network with residual
    let normed2 = residual1.layer_norm(ln2_gamma, ln2_beta, 1e-5);
    let ffn_hidden = normed2.matmul(ffn_w1).add_bias(ffn_b1).gelu(); 
    // Expand + activate
    let ffn_out = ffn_hidden.matmul(ffn_w2).add_bias(ffn_b2);          // Project back

    let residual2 = residual1.add(&ffn_out);  // Skip connection
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
) -> Tensor {
    // Add this for profiling GPT2 forward pass
    let total_guard = crate::profiler::ProfileGuard::new("gpt2_forward_total");

    let tok_emb = embedding_lookup(wte, token_ids);
    let positions: Vec<usize> = (0..token_ids.len()).collect();
    let pos_emb = embedding_lookup(wpe, &positions);
    let mut hidden = tok_emb.add(&pos_emb);
    
    for block in blocks {
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
        );
    }
    
    let normed = hidden.layer_norm(ln_f_gamma, ln_f_beta, 1e-5);

    // Profiling GPT2 Forward Pass
    crate::profiler::print_profile_summary();

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