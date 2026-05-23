use core::model::{ModelLoader, safetensors::SafeTensorsLoader};
use core::{Tensor, split_qkv, gpt2_forward};

fn main() {
    println!("========== Loading GPT-2 Model ==========\n");

    // 1. Load the safetensors file
    let model = SafeTensorsLoader::load("models/gpt2/model.safetensors")
        .expect("Failed to load model. Run: hf download openai-community/gpt2 --local-dir ./models/gpt2");

    println!("Loaded {} tensors\n", model.tensors.len());

    // 2. Extract global tensors
    let wte = model.tensors.get("wte.weight")
        .expect("Missing wte.weight");
    let wpe = model.tensors.get("wpe.weight")
        .expect("Missing wpe.weight");
    let ln_f_gamma = model.tensors.get("ln_f.weight")
        .expect("Missing ln_f.weight");
    let ln_f_beta = model.tensors.get("ln_f.bias")
        .expect("Missing ln_f.bias");

    println!("wte.weight: {:?}", wte.shape);
    println!("wpe.weight: {:?}", wpe.shape);
    println!("ln_f.weight: {:?}", ln_f_gamma.shape);
    println!("ln_f.bias: {:?}", ln_f_beta.shape);
    println!();

    // 3. Build the 12 transformer blocks from fused QKV weights
    let mut blocks: Vec<(
        Tensor, Tensor,  // q_w, q_b
        Tensor, Tensor,  // k_w, k_b
        Tensor, Tensor,  // v_w, v_b
        Tensor, Tensor,  // out_w, out_b
        Tensor, Tensor,  // ln1_gamma, ln1_beta
        Tensor, Tensor,  // ffn_w1, ffn_b1
        Tensor, Tensor,  // ffn_w2, ffn_b2
        Tensor, Tensor,  // ln2_gamma, ln2_beta
    )> = Vec::new();

    for layer_idx in 0..12 {
        // Get fused QKV weight and bias
        let c_attn_weight = model.tensors.get(&format!("h.{}.attn.c_attn.weight", layer_idx))
            .expect(&format!("Missing h.{}.attn.c_attn.weight", layer_idx));
        let c_attn_bias = model.tensors.get(&format!("h.{}.attn.c_attn.bias", layer_idx))
            .expect(&format!("Missing h.{}.attn.c_attn.bias", layer_idx));

        // Split into Q, K, V
        let (q_w, q_b, k_w, k_b, v_w, v_b) = split_qkv(c_attn_weight, c_attn_bias);

        // Output projection
        let out_w = model.tensors.get(&format!("h.{}.attn.c_proj.weight", layer_idx)).unwrap();
        let out_b = model.tensors.get(&format!("h.{}.attn.c_proj.bias", layer_idx)).unwrap();

        // LayerNorm 1
        let ln1_g = model.tensors.get(&format!("h.{}.ln_1.weight", layer_idx)).unwrap();
        let ln1_b = model.tensors.get(&format!("h.{}.ln_1.bias", layer_idx)).unwrap();

        // FFN
        let ffn_w1 = model.tensors.get(&format!("h.{}.mlp.c_fc.weight", layer_idx)).unwrap();
        let ffn_b1 = model.tensors.get(&format!("h.{}.mlp.c_fc.bias", layer_idx)).unwrap();
        let ffn_w2 = model.tensors.get(&format!("h.{}.mlp.c_proj.weight", layer_idx)).unwrap();
        let ffn_b2 = model.tensors.get(&format!("h.{}.mlp.c_proj.bias", layer_idx)).unwrap();

        // LayerNorm 2
        let ln2_g = model.tensors.get(&format!("h.{}.ln_2.weight", layer_idx)).unwrap();
        let ln2_b = model.tensors.get(&format!("h.{}.ln_2.bias", layer_idx)).unwrap();
        println!("Layer {}: QKV split done (d_model={})", layer_idx, q_w.shape[0]);
        blocks.push((
            q_w, q_b, k_w, k_b, v_w, v_b,
            out_w.clone(), out_b.clone(),
            ln1_g.clone(), ln1_b.clone(),
            ffn_w1.clone(), ffn_b1.clone(),
            ffn_w2.clone(), ffn_b2.clone(),
            ln2_g.clone(), ln2_b.clone(),
        ));

        
    }

    println!("\n========== Running Forward Pass ==========\n");

    // 4. Run a forward pass with a simple prompt
    // Token IDs for: "Hello, world!" (approximate, we'll use simple IDs)
    // GPT-2 tokenizer: "The" = 464, " quick" = 2068, " brown" = 7586
    // Let's use a simple sequence: "The cat sat on the mat"
    let token_ids = vec![464, 3797, 3332, 319, 262, 2603]; // "The cat sat on the mat"
    
    println!("Input token IDs: {:?}", token_ids);
    println!("Sequence length: {}\n", token_ids.len());

    let logits = gpt2_forward(
        &token_ids,
        wte,
        wpe,
        &blocks,
        ln_f_gamma,
        ln_f_beta,
        wte,  // GPT-2 ties embedding and LM head weights
    );

    println!("Output logits shape: {:?}", logits.shape);
    
    // 5. Get the predicted next token (argmax of last position)
    let seq_len = token_ids.len();
    let vocab_size = logits.shape[1];
    let last_pos_start = (seq_len - 1) * vocab_size;
    let last_logits = &logits.data[last_pos_start..last_pos_start + vocab_size];
    
    // Find argmax
    let mut max_val = f32::NEG_INFINITY;
    let mut max_idx = 0;
    for (i, &val) in last_logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    
    println!("Predicted next token ID: {}", max_idx);
    println!("Prediction confidence (logit): {:.4}", max_val);
    
    // Print top 5 predictions
    let mut indexed: Vec<(usize, f32)> = last_logits.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nTop 5 next token predictions:");
    for (i, (token_id, logit)) in indexed.iter().take(5).enumerate() {
        println!("  {}. Token ID {}: {:.4}", i + 1, token_id, logit);
    }

    println!("\n✅ GPT-2 forward pass complete!");
}