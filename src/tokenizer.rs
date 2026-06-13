use anyhow::Result;
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use std::fs;

#[derive(Serialize, Deserialize)]
pub struct BPETokenizer {
    pub stoi: HashMap<String, usize>,
    pub itos: Vec<String>,
}

impl BPETokenizer {
    // Build a very simple char+BPE-like tokenizer from corpus:
    // - start from characters, then learn merges up to vocab_size.
    pub fn train_from_text(text: &str, vocab_size: usize) -> Self {
        // initialize with characters
        let mut freq: HashMap<Vec<String>, usize> = HashMap::new();
        for word in text.split_whitespace() {
            let symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            let mut seq = symbols.clone();
            seq.push("</w>".to_string());
            *freq.entry(seq).or_insert(0) += 1;
        }
        // gather vocab set
        let mut merges: Vec<(String, String)> = Vec::new();
        let mut vocab: HashSet<String> = HashSet::new();
        for ch in text.chars() {
            vocab.insert(ch.to_string());
        }
        vocab.insert("</w>".to_string());
        // naive merge loop
        while vocab.len() < vocab_size {
            // count pair frequencies
            let mut pairs: HashMap<(String,String), usize> = HashMap::new();
            for (seq, &count) in freq.iter() {
                for i in 0..seq.len().saturating_sub(1) {
                    let a = seq[i].clone();
                    let b = seq[i+1].clone();
                    *pairs.entry((a,b)).or_insert(0) += count;
                }
            }
            if pairs.is_empty() { break; }
            // find best pair
            let best = pairs.iter().max_by_key(|(_,c)| *c).map(|(k,_)| k.clone());
            if best.is_none() { break; }
            let best = best.unwrap();
            merges.push((best.0.clone(), best.1.clone()));
            // apply merge
            let merge_token = format!("{}{}", best.0, best.1);
            let mut new_freq = HashMap::new();
            for (seq, &count) in freq.iter() {
                let mut new_seq: Vec<String> = Vec::new();
                let mut i = 0;
                while i < seq.len() {
                    if i+1 < seq.len() && seq[i]==best.0 && seq[i+1]==best.1 {
                        new_seq.push(merge_token.clone());
                        i += 2;
                    } else {
                        new_seq.push(seq[i].clone());
                        i += 1;
                    }
                }
                *new_freq.entry(new_seq).or_insert(0) += count;
            }
            freq = new_freq;
            vocab.insert(merge_token);
        }
        // assemble itos
        let mut itos: Vec<String> = vocab.into_iter().collect();
        itos.sort();
        let stoi: HashMap<String, usize> = itos.iter().enumerate().map(|(i,t)| (t.clone(), i)).collect();
        BPETokenizer { stoi, itos }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let s = serde_json::to_string(self)?;
        fs::write(path, s)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let s = fs::read_to_string(path)?;
        let t: BPETokenizer = serde_json::from_str(&s)?;
        Ok(t)
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        // greedy longest-match
        let mut ids = Vec::new();
        for token in text.split_whitespace() {
            let mut seq: Vec<String> = token.chars().map(|c| c.to_string()).collect();
            seq.push("</w>".to_string());
            let mut i = 0;
            while i < seq.len() {
                // try longest match up to remaining length
                let mut matched = None;
                for j in (i+1..=seq.len()).rev() {
                    let piece = seq[i..j].concat();
                    if let Some(&id) = self.stoi.get(&piece) {
                        matched = Some((piece, id, j));
                        break;
                    }
                }
                if let Some((_piece, id, j)) = matched {
                    ids.push(id);
                    i = j;
                } else {
                    // fallback to single char (should exist)
                    let ch = seq[i].clone();
                    if let Some(&id) = self.stoi.get(&ch) {
                        ids.push(id);
                    }
                    i += 1;
                }
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids {
            let tok = &self.itos[id];
            if tok == "</w>" { out.push(' '); continue; }
            out.push_str(tok);
        }
        out.trim().to_string()
    }
}

