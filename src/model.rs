use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::nn::{Module, Linear, Embedding, Dropout};
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig};
use crate::config::ModelConfig;
use burn::record::{Record, Registrable};

pub struct StoryModel<B: Backend> {
    pub embed: Embedding<B>,
    pub transformer: TransformerEncoder<B>,
    pub lm_head: Linear<B>,
    pub dropout: Dropout<B>,
}

impl<B: Backend> StoryModel<B> {
    pub fn new(config: &ModelConfig) -> Self {
        let embed = Embedding::new(config.vocab_size, config.d_model);
        let transformer_cfg = TransformerEncoderConfig::new(
            config.d_model,
            config.n_heads,
            config.d_ff,
            config.n_layers
        );
        let transformer = TransformerEncoder::new(transformer_cfg);
        let lm_head = Linear::new(config.d_model, config.vocab_size);
        let dropout = Dropout::new(config.dropout);
        Self { embed, transformer, lm_head, dropout }
    }
}

impl<B: Backend> Module for StoryModel<B> {
    type Input = Tensor<B, 2>;
    type Output = Tensor<B, 3>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        // input shape: (batch, seq)
        let x = self.embed.forward(input);
        let x = self.dropout.forward(x);
        let x = self.transformer.forward(x);
        let logits = self.lm_head.forward(x);
        logits
    }
}

