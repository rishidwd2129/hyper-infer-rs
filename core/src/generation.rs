use crate::Tensor;
use crate::gpt2_forward;
// 1. ADDED: Import our new KVCache
use crate::kv_cache::KVCache;
use rand::Rng;
use rand::RngExt;

// Define a type alias for the massive 16-tuple layer block weights 
// to avoid repeating this long signature across your functions.
pub type Gpt2BlockWeights = (
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, 
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
);

pub struct GenerationEngine<'a> {
    wte: &'a Tensor,
    wpe: &'a Tensor,
    blocks: &'a [(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)],
    ln_f_gamma: &'a Tensor,
    ln_f_beta: &'a Tensor,
}

impl<'a> GenerationEngine<'a> {
    // Constructor
    pub fn new(
        wte: &'a Tensor,
        wpe: &'a Tensor,
        blocks: &'a [(Gpt2BlockWeights)],
        ln_f_gamma: &'a Tensor,
        ln_f_beta: &'a Tensor,
    ) -> Self {
        Self { wte, wpe, blocks, ln_f_gamma, ln_f_beta }
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt_tokens: &[usize],
        max_tokens: usize,
        temperature: f32, 
        top_k: usize,   
        mut on_token: F,
    ) where
        F: FnMut(usize) -> bool,
    {
        let _gen_guard = crate::profiler::ProfileGuard::new("full_generation_loop");

        // 2. ADDED: Initialize the KV Cache for GPT-2 Small
        // 12 layers, 12 heads, head_dim of 64 (768 d_model / 12), and max context of 1024
        let mut kv_cache = KVCache::new(12, 12, 64, 1024);
        //Updated memory allocator
        // 👈 Single allocation for the entire generation run
    // GPT-2 Small: max_seq_len=1024, d_model=768
        let mut workspace = crate::ComputeWorkspace::new(1024, 768);

        // We clone the prompt into our context window to keep track of the full text
        let mut context = prompt_tokens.to_vec();

        // 3. ADDED: The dynamic input buffer. 
        // For the very first pass (Prefill), we feed the entire prompt.
        let mut input_tokens = prompt_tokens.to_vec();

        for _ in 0..max_tokens {
            // Run the forward pass. 
            // - Loop 1: input_tokens length is N (Prefill)
            // - Loop 2+: input_tokens length is exactly 1 (Decode)
            let logits = gpt2_forward(
                &input_tokens,
                self.wte, self.wpe, self.blocks, self.ln_f_gamma, self.ln_f_beta, self.wte,
                &mut kv_cache, // 👈 Pass our new cache into the forward pass!
                &mut workspace,
            );

            // Extract the logits for the very last token in the sequence we just processed
            let vocab_size = logits.shape[1];
            let seq_len = input_tokens.len();
            let last_pos_start = (seq_len - 1) * vocab_size;
            let last_logits = &logits.data[last_pos_start..last_pos_start + vocab_size];

            // Sample the next token ID
            let next_token_id = self.sample(last_logits, temperature, top_k);

            // Append to context for keeping track of the final output
            context.push(next_token_id);

            // Fire the callback to print the word to the terminal
            if !on_token(next_token_id) {
                break;
            }

            // 4. ADDED: The Decode Phase Transition!
            // Instead of appending the new token to the prompt and feeding the whole 
            // thing back in, we replace the input buffer with ONLY the single new token.
            // The model will remember the rest because it's stored in `kv_cache`.
            input_tokens = vec![next_token_id];
            
        } // <-- Loop ends here

        println!("\n\n========== GENERATION COMPLETE ==========");
        crate::profiler::print_profile_summary();
    }

    fn sample_greedy(&self, logits: &[f32]) -> usize {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        
        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    }

    /// A robust sampler featuring Temperature and Top-K filtering
    fn sample(&self, logits: &[f32], temperature: f32, top_k: usize) -> usize {
        if temperature == 0.0 {
            return self.sample_greedy(logits);
        }

        let mut adjusted_logits: Vec<f32> = logits.iter()
            .map(|&l| l / temperature)
            .collect();

        if top_k > 0 && top_k < adjusted_logits.len() {
            let mut indexed_logits: Vec<(usize, f32)> = adjusted_logits
                .iter()
                .enumerate()
                .map(|(i, &val)| (i, val))
                .collect();

            indexed_logits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let cutoff_value = indexed_logits[top_k - 1].1;

            for val in adjusted_logits.iter_mut() {
                if *val < cutoff_value {
                    *val = f32::NEG_INFINITY;
                }
            }
        }

        let max_logit = adjusted_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = adjusted_logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exps: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exps).collect();

        let mut rng = rand::rng();
        let p: f32 = rng.random(); 

        let mut cumulative = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if p <= cumulative {
                return i;
            }
        }

        probs.len() - 1
    }
}