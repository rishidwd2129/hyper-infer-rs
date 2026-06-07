// src/kv_cache.rs

/// Stores accumulated K and V vectors for ONE transformer layer.
/// Uses a pre-allocated flat 1D memory arena to prevent dynamic heap allocations,
/// while logically grouping data per-head to optimize attention loop reads.
pub struct LayerKVCache {
    /// Flat arena. Layout: [head][pos][dim]
    pub k_data: Vec<f32>,
    pub v_data: Vec<f32>,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub current_seq_len: usize,
}

impl LayerKVCache {
    pub fn new(num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let total_size = num_heads * max_seq_len * head_dim;
        
        Self {
            k_data: vec![0.0; total_size],
            v_data: vec![0.0; total_size],
            num_heads,
            head_dim,
            max_seq_len,
            current_seq_len: 0,
        }
    }

    /// How many token positions are currently cached?
    pub fn seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// Appends one new K vector for head `h` using zero-allocation in-place mutation.
    pub fn push_k(&mut self, h: usize, vec: &[f32]) {
        debug_assert_eq!(vec.len(), self.head_dim);
        
        // Calculate where this head's block starts in the flat array
        let head_offset = h * (self.max_seq_len * self.head_dim);
        // Calculate where the new token belongs within that head's block
        let write_start = head_offset + (self.current_seq_len * self.head_dim);
        
        self.k_data[write_start..write_start + self.head_dim].copy_from_slice(vec);
    }

    pub fn push_v(&mut self, h: usize, vec: &[f32]) {
        debug_assert_eq!(vec.len(), self.head_dim);
        
        let head_offset = h * (self.max_seq_len * self.head_dim);
        let write_start = head_offset + (self.current_seq_len * self.head_dim);
        
        self.v_data[write_start..write_start + self.head_dim].copy_from_slice(vec);
    }

    /// Increments the sequence length counter. Call this ONLY ONCE per token step, 
    /// after all heads have pushed their K and V data.
    pub fn increment_seq_len(&mut self) {
        if self.current_seq_len < self.max_seq_len {
            self.current_seq_len += 1;
        }
    }

    /// Returns a slice of shape [current_seq_len * head_dim] for head `h`.
    /// This represents all historical keys for this specific head.
    pub fn k_slice(&self, h: usize) -> &[f32] {
        let head_offset = h * (self.max_seq_len * self.head_dim);
        let active_length = self.current_seq_len * self.head_dim;
        &self.k_data[head_offset..head_offset + active_length]
    }

    pub fn v_slice(&self, h: usize) -> &[f32] {
        let head_offset = h * (self.max_seq_len * self.head_dim);
        let active_length = self.current_seq_len * self.head_dim;
        &self.v_data[head_offset..head_offset + active_length]
    }

    pub fn clear(&mut self) {
        self.current_seq_len = 0;
        // No need to physically write 0.0 to memory. Resetting the counter 
        // means we will just safely overwrite the old data on the next run!
    }
}

/// One LayerKVCache per transformer block.
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerKVCache::new(num_heads, head_dim, max_seq_len))
                .collect(),
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }
}