use anyhow::Result;
use crate::tokenizer::BPETokenizer;
use crate::config::ModelConfig;
use crate::model::StoryModel;
use burn::tensor::NdArrayBackend;
use burn::tensor::Tensor;
use rand::Rng;

pub fn nucleus_sampling(probs: &Vec<f32>, top_p: f32) -> usize {
    // probs: probability distribution over vocab
    let mut pairs: Vec<(usize,f32)> = probs.iter().enumerate().map(|(i,&p)|(i,p)).collect();
    pairs.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
    let mut cum = 0.0;
    let mut cutoff = 0usize;
    for (i, &(_, p)) in pairs.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cutoff = i;
            break;
        }
    }
    let slice = &pairs[..=cutoff];
    let sum: f32 = slice.iter().map(|x| x.1).sum();
    let mut rng = rand::thread_rng();
    let mut r: f32 = rng.gen::<f32>() * sum;
    for &(idx, p) in slice {
        if r <= p {
            return idx;
        }
        r -= p;
    }
    slice.last().unwrap().0
}

pub fn generate(corpus_path: &str, tokenizer_path: &str, checkpoint: &str, prompt: &str, length: usize, temperature: f32, top_p: f32) -> Result<String> {
    let backend = NdArrayBackend::default();
    let cfg = ModelConfig::default();
    let tokenizer = BPETokenizer::load(tokenizer_path)?;
    let mut model = StoryModel::<NdArrayBackend>::new(&cfg);
    // NOTE: loading parameters from checkpoint not implemented (placeholder)
    // Encode prompt
    let mut ids = tokenizer.encode(prompt);
    for _ in 0..length {
        // build input tensor (1, seq)
        let input_ids: Vec<i64> = ids.iter().rev().take(cfg.max_len).cloned().collect::<Vec<_>>().into_iter().rev().map(|x| x as i64).collect();
        let seq = input_ids.len();
        let input = Tensor::from_data(input_ids.clone(), vec![1usize, seq]);
        let logits = model.forward(input);
        // take last token logits
        // Convert logits tensor to Vec<f32> (placeholder)
        // For demo, generate random token
        let mut rng = rand::thread_rng();
        let next = rng.gen_range(0..tokenizer.itos.len());
        ids.push(next);
    }
    Ok(tokenizer.decode(&ids))
}

