mod lib;
use clap::{Parser, Subcommand};
use anyhow::Result;
use crate::config::ModelConfig;

#[derive(Parser)]
#[clap(name = "rus_story_transformer")]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        corpus: String,
        tokenizer: String,
        #[clap(long, default_value = "32000")]
        vocab: usize,
        #[clap(long, default_value = "6")]
        layers: usize,
        #[clap(long, default_value = "512")]
        dmodel: usize,
        #[clap(long, default_value = "8")]
        heads: usize,
        #[clap(long, default_value = "10")]
        epochs: usize,
    },
    Generate {
        corpus: String,
        tokenizer: String,
        checkpoint: String,
        prompt: String,
        #[clap(long, default_value = "300")]
        length: usize,
        #[clap(long, default_value = "1.0")]
        temp: f32,
        #[clap(long, default_value = "0.9")]
        top_p: f32,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train { corpus, tokenizer, vocab, layers, dmodel, heads, epochs } => {
            let mut cfg = ModelConfig::default();
            cfg.vocab_size = vocab;
            cfg.n_layers = layers;
            cfg.d_model = dmodel;
            cfg.n_heads = heads;
            crate::train::train(&corpus, &tokenizer, cfg, epochs)?;
        }
        Commands::Generate { corpus, tokenizer, checkpoint, prompt, length, temp, top_p } => {
            let out = crate::generate::generate(&corpus, &tokenizer, &checkpoint, &prompt, length, temp, top_p)?;
            println!("{}", out);
        }
    }
    Ok(())
}

