use core::model::{ModelLoader, safetensors::SafeTensorsLoader};
use core::{Tensor, split_qkv, gpt2_forward};
use core::generation::GenerationEngine;
use std::io::{self, Write};
// 1. Import the Tokenizer
use tokenizers::Tokenizer;

fn main() {
    println!("========== Loading GPT-2 Model ==========\n");

    let model = SafeTensorsLoader::load("models/gpt2/model.safetensors")
        .expect("Failed to load model. Run: hf download openai-community/gpt2 --local-dir ./models/gpt2");

    // 2. Load the Tokenizer
    let tokenizer = Tokenizer::from_file("models/gpt2/tokenizer.json")
        .expect("Failed to load tokenizer. Please download tokenizer.json to your models/gpt2 directory.");

    println!("Loaded {} tensors\n", model.tensors.len());

    let wte = model.tensors.get("wte.weight").expect("Missing wte.weight");
    let wpe = model.tensors.get("wpe.weight").expect("Missing wpe.weight");
    let ln_f_gamma = model.tensors.get("ln_f.weight").expect("Missing ln_f.weight");
    let ln_f_beta = model.tensors.get("ln_f.bias").expect("Missing ln_f.bias");

    let mut blocks = Vec::new();

    for layer_idx in 0..12 {
        let c_attn_weight = model.tensors.get(&format!("h.{}.attn.c_attn.weight", layer_idx)).unwrap();
        let c_attn_bias = model.tensors.get(&format!("h.{}.attn.c_attn.bias", layer_idx)).unwrap();

        let (q_w, q_b, k_w, k_b, v_w, v_b) = split_qkv(c_attn_weight, c_attn_bias);

        let out_w = model.tensors.get(&format!("h.{}.attn.c_proj.weight", layer_idx)).unwrap();
        let out_b = model.tensors.get(&format!("h.{}.attn.c_proj.bias", layer_idx)).unwrap();
        let ln1_g = model.tensors.get(&format!("h.{}.ln_1.weight", layer_idx)).unwrap();
        let ln1_b = model.tensors.get(&format!("h.{}.ln_1.bias", layer_idx)).unwrap();
        let ffn_w1 = model.tensors.get(&format!("h.{}.mlp.c_fc.weight", layer_idx)).unwrap();
        let ffn_b1 = model.tensors.get(&format!("h.{}.mlp.c_fc.bias", layer_idx)).unwrap();
        let ffn_w2 = model.tensors.get(&format!("h.{}.mlp.c_proj.weight", layer_idx)).unwrap();
        let ffn_b2 = model.tensors.get(&format!("h.{}.mlp.c_proj.bias", layer_idx)).unwrap();
        let ln2_g = model.tensors.get(&format!("h.{}.ln_2.weight", layer_idx)).unwrap();
        let ln2_b = model.tensors.get(&format!("h.{}.ln_2.bias", layer_idx)).unwrap();
        
        blocks.push((
            q_w, q_b, k_w, k_b, v_w, v_b,
            out_w.clone(), out_b.clone(),
            ln1_g.clone(), ln1_b.clone(),
            ffn_w1.clone(), ffn_b1.clone(),
            ffn_w2.clone(), ffn_b2.clone(),
            ln2_g.clone(), ln2_b.clone(),
        ));
    }

    println!("\n========== Running Generation Loop ==========\n");

    // 3. Dynamically encode your prompt instead of hardcoding IDs!
    let prompt = "The cat sat on the mat";
    let encoding = tokenizer.encode(prompt, false).unwrap();
    
    // Convert u32 IDs from the tokenizer into the usize IDs your engine expects
    let token_ids: Vec<usize> = encoding.get_ids().iter().map(|&id| id as usize).collect();

    let mut engine = GenerationEngine::new(
        wte, 
        wpe, 
        &blocks, 
        ln_f_gamma, 
        ln_f_beta
    );

    print!("{}", prompt); // Print the initial prompt so the output flows naturally
    io::stdout().flush().unwrap();

    // 4. Decode the stream as it generates
    // Added Temperature (0.7) and Top-K (40) arguments here
    engine.generate_stream(&token_ids, 30, 0.7, 40, |new_token_id| {
        // Decode the single predicted token ID back into a string
        let word = tokenizer.decode(&[new_token_id as u32], false).unwrap_or_default();
        
        print!("{}", word);
        io::stdout().flush().unwrap(); 
        
        true 
    });

    println!("\n\n✅ Generation stream completed successfully.");
}