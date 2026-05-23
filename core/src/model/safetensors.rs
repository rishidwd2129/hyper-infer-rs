use std::collections::HashMap;
use std::fs;
use std::io::Read;
use serde::Deserialize;
use serde_json;
use crate::Tensor;
use super::{ModelData, ModelError, ModelLoader};

#[derive(Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct SafeTensorsLoader;

impl ModelLoader for SafeTensorsLoader {
    fn load(path: &str) -> Result<ModelData, ModelError> {
        // 1. Read the entire file into memory
        let mut file = fs::File::open(path)
            .map_err(|_| ModelError::FileNotFound(path.to_string()))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;

        // 2. Parse the 8-byte header length
        if buffer.len() < 8 {
            return Err(ModelError::InvalidHeader("File too small".into()));
        }
        
        let header_len_bytes: [u8; 8] = buffer[0..8].try_into().unwrap();
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        // 3. Extract and parse the JSON header
        let header_start = 8;
        let header_end = header_start + header_len;
        
        if buffer.len() < header_end {
            return Err(ModelError::InvalidHeader("Header extends past file".into()));
        }

        let header_json = std::str::from_utf8(&buffer[header_start..header_end])
            .map_err(|e| ModelError::InvalidHeader(e.to_string()))?;

        // *** THE FIX: Parse to generic JSON first, then filter ***
        let raw_map: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)
            .map_err(|e| ModelError::InvalidHeader(format!("JSON parse error: {}", e)))?;

        let data_start = header_end;
        let data_blob = &buffer[data_start..];
        let mut tensors = HashMap::new();

        for (name, value) in raw_map {
            // Skip metadata entries that don't have tensor info
            if name == "__metadata__" {
                continue;
            }

            // Convert the generic JSON value to TensorInfo
            let info: TensorInfo = serde_json::from_value(value)
                .map_err(|e| ModelError::InvalidHeader(
                    format!("Failed to parse tensor '{}': {}", name, e)
                ))?;

            // Verify dtype
            if info.dtype != "F32" {
                return Err(ModelError::InvalidFormat(
                    format!("Unsupported dtype '{}' for tensor '{}'. Only F32 is supported.", 
                            info.dtype, name)
                ));
            }

            // Extract bytes
            let start = info.data_offsets[0];
            let end = info.data_offsets[1];
            
            if end > data_blob.len() || start > end {
                return Err(ModelError::InvalidFormat(
                    format!("Invalid data offsets for tensor '{}'", name)
                ));
            }

            let raw_bytes = &data_blob[start..end];

            // Convert to f32
            let float_count = raw_bytes.len() / 4;
            let mut data = Vec::with_capacity(float_count);
            
            for chunk in raw_bytes.chunks_exact(4) {
                let bytes: [u8; 4] = chunk.try_into().unwrap();
                data.push(f32::from_le_bytes(bytes));
            }

            // Verify size
            let expected_size: usize = info.shape.iter().product();
            if data.len() != expected_size {
                return Err(ModelError::TensorShapeMismatch {
                    name: name.clone(),
                    expected: expected_size,
                    got: data.len(),
                });
            }

            tensors.insert(name, Tensor::new(data, info.shape));
        }

        Ok(ModelData { tensors })
    }
}