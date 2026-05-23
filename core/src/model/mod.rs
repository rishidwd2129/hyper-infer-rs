pub mod safetensors;
pub mod gguf;  // Future

use std::collections::HashMap;
use crate::Tensor;

/// Represents a loaded model: named tensors with their data and shapes
pub struct ModelData {
    pub tensors: HashMap<String, Tensor>,
}

/// Common interface for all model loaders
pub trait ModelLoader {
    /// Load model from a file path, returning all named tensors
    fn load(path: &str) -> Result<ModelData, ModelError>;
}

#[derive(Debug)]
pub enum ModelError {
    FileNotFound(String),
    InvalidFormat(String),
    InvalidHeader(String),
    TensorShapeMismatch { name: String, expected: usize, got: usize },
}