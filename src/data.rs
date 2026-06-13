use crate::tokenizer::BPETokenizer;
use anyhow::Result;
use std::fs;
use rand::seq::SliceRandom;

pub fn load_corpus(path: &str) -> Result<String> {
    let s = fs::read_to_string(path)?;
    Ok(s)
}

pub fn build_tokenizer(corpus: &str, vocab_size: usize, out_path: &str) -> Result<BPETokenizer> {
    let tok = BPETokenizer::train_from_text(corpus, vocab_size);
    tok.save(out_path)?;
    Ok(tok)
}

pub fn create_batches(ids: &[usize], seq_len: usize, batch_size: usize) -> Vec<Vec<usize>> {
    let step = seq_len;
    let mut chunks = Vec::new();
    let mut i = 0;
    while i + seq_len < ids.len() {
        let mut batch = Vec::new();
        for b in 0..batch_size {
            let start = i + b * step;
            if start + seq_len >= ids.len() { break; }
            batch.extend_from_slice(&ids[start..start+seq_len]);
        }
        if !batch.is_empty() { chunks.push(batch); }
        i += seq_len * batch_size;
    }
    chunks
}

