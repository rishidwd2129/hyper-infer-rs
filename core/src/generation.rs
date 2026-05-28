use crate::Tensor;
use crate::gpt2_forward;
use rand::Rng;
use rand::RngExt;

// Define a type alias for the massive 16-tuple layer block weights 
// to avoid repeating this long signature across your functions.
pub type Gpt2BlockWeights = (
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, 
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
);

pub struct GenerationEngine<'a> {
    // We use references so the engine doesn't duplicate the massive weight matrices
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
        temperature: f32, // <-- ADDED
        top_k: usize,   
        mut on_token: F,
    ) where
        F: FnMut(usize) -> bool,
    {
        // Clone the prompt into our working context window
        let mut context = prompt_tokens.to_vec();

        // 1. ADD THIS: Profile the entire generation loop end-to-end
        let _gen_guard = crate::profiler::ProfileGuard::new("full_generation_loop");

        for _ in 0..max_tokens {
            // 1. Run the forward pass
            let logits = gpt2_forward(
                &context,
                self.wte, self.wpe, self.blocks, self.ln_f_gamma, self.ln_f_beta, self.wte,
            );

            // 2. Extract the logits for the very last token
            let vocab_size = logits.shape[1];
            let seq_len = context.len();
            let last_pos_start = (seq_len - 1) * vocab_size;
            let last_logits = &logits.data[last_pos_start..last_pos_start + vocab_size];

            // 3. Sample the next token (Greedy Decoding for now)
            let next_token_id = self.sample(last_logits,temperature, top_k);

            // 4. Append to context for the next iteration
            context.push(next_token_id);

            // 5. Fire the callback. Break if the UI says stop.
            if !on_token(next_token_id) {
                break;
            }
        } // <-- Loop ends here

        // 2. ADD THIS: Print the accumulated summary exactly ONCE
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
        // If temperature is 0, fallback to pure deterministic greedy decoding
        if temperature == 0.0 {
            return self.sample_greedy(logits);
        }

        // 1. Apply Temperature (higher = more random, lower = more focused)
        let mut adjusted_logits: Vec<f32> = logits.iter()
            .map(|&l| l / temperature)
            .collect();

        // 2. Apply Top-K Filtering
        if top_k > 0 && top_k < adjusted_logits.len() {
            // Pair each logit with its vocabulary index so we can sort them safely
            let mut indexed_logits: Vec<(usize, f32)> = adjusted_logits
                .iter()
                .enumerate()
                .map(|(i, &val)| (i, val))
                .collect();

            // Sort descending by logit value. (Rust f32 doesn't implement Ord due to NaN, so we unwrap)
            indexed_logits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Find the cutoff threshold (the value of the Kth best token)
            let cutoff_value = indexed_logits[top_k - 1].1;

            // Mask out anything below the cutoff with -Infinity so it becomes 0% probability
            for val in adjusted_logits.iter_mut() {
                if *val < cutoff_value {
                    *val = f32::NEG_INFINITY;
                }
            }
        }

        // 3. Softmax (Convert logits to probabilities that sum to 1.0)
        let max_logit = adjusted_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = adjusted_logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exps: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exps).collect();

        // 4. Sample from the distribution
        let mut rng = rand::rng();
        let p: f32 = rng.random(); // Random float between 0.0 and 1.0

        let mut cumulative = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if p <= cumulative {
                return i;
            }
        }

        // Fallback in case of float rounding errors
        probs.len() - 1
    }
}