use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_len: usize,
    pub dropout: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 512,
            n_heads: 8,
            n_layers: 6,
            d_ff: 2048,
            max_len: 1024,
            dropout: 0.1,
        }
    }
}

