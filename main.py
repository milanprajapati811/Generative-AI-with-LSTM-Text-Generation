
import os
import sys
import random
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import run_preprocessing
from model      import build_model, build_shallow_model, build_deep_model
from train      import train_model, load_best_model, plot_training_history, compare_architectures
from generate   import generate_all_samples


CONFIG = {

    "max_chars"       : 500_000,   
    "seq_length"      : 30,        

   
    "embedding_dim"   : 128,       
    "lstm_units_1"    : 256,       
    "lstm_units_2"    : 128,       
    "dropout_rate"    : 0.3,       
    "learning_rate"   : 1e-3,      

   
    "epochs"          : 30,        
    "batch_size"      : 256,      
    "val_split"       : 0.1,       

    
    "checkpoint_path" : "models/best_model.keras",
    "history_plot"    : "outputs/training_history.png",
    "samples_path"    : "outputs/generated_samples.txt",

    
    "gen_num_words"   : 80,
    "gen_seeds"       : [
        "to be or not to be",
        "shall i compare thee to",
        "all the worlds a stage",
        "what light through yonder window",
        "friends romans countrymen lend me",
    ],
    "gen_temperatures": [0.5, 0.8, 1.2],

    
    "run_bonus"       : True,     
    "bonus_epochs"    : 3,        
    "bonus_samples"   : 50_000,    

    
    "seed"            : 42,
}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_banner(text: str) -> None:
    print("\n" + "="*65)
    print(f"  {text}")
    print("="*65)


def main():
    set_seeds(CONFIG["seed"])
    print("\n" + "="*65)
    print("  🎭 LSTM TEXT GENERATION — SHAKESPEARE'S WORKS")
    print("  Dataset : Project Gutenberg Complete Works")
    print("  URL     : https://www.gutenberg.org/files/100/100-0.txt")
    print("="*65)

    print_banner("STEP 1 — DATA PREPROCESSING")

    X, y, word2idx, idx2word, vocab_size = run_preprocessing(
        seq_length = CONFIG["seq_length"],
        max_chars  = CONFIG["max_chars"],
    )

    print(f"\n  Vocabulary size : {vocab_size:,}")
    print(f"  Total sequences : {len(X):,}")
    print(f"  X shape         : {X.shape}")
    print(f"  y shape         : {y.shape}")

    print_banner("STEP 2 — MODEL ARCHITECTURE")

    model = build_model(
        vocab_size    = vocab_size,
        seq_length    = CONFIG["seq_length"],
        embedding_dim = CONFIG["embedding_dim"],
        lstm_units_1  = CONFIG["lstm_units_1"],
        lstm_units_2  = CONFIG["lstm_units_2"],
        dropout_rate  = CONFIG["dropout_rate"],
        learning_rate = CONFIG["learning_rate"],
    )

    print_banner("STEP 3 — TRAINING")

    history = train_model(
        model           = model,
        X               = X,
        y               = y,
        epochs          = CONFIG["epochs"],
        batch_size      = CONFIG["batch_size"],
        val_split       = CONFIG["val_split"],
        checkpoint_path = CONFIG["checkpoint_path"],
    )

    
    plot_training_history(history, save_path=CONFIG["history_plot"])

    print_banner("STEP 4 — LOADING BEST MODEL")

    best_model = load_best_model(CONFIG["checkpoint_path"])
    if best_model is None:
        print("  Using last epoch model (no checkpoint found)")
        best_model = model

    print_banner("STEP 5 — TEXT GENERATION")

    generate_all_samples(
        model        = best_model,
        word2idx     = word2idx,
        idx2word     = idx2word,
        seq_length   = CONFIG["seq_length"],
        seeds        = CONFIG["gen_seeds"],
        temperatures = CONFIG["gen_temperatures"],
        num_words    = CONFIG["gen_num_words"],
        save_path    = CONFIG["samples_path"],
    )


    if CONFIG["run_bonus"]:
        print_banner("STEP 6 (BONUS) — ARCHITECTURE COMPARISON")

        arch_models = {
            "ShallowLSTM (1 layer,  emb=64,  lstm=128)"      : build_shallow_model(vocab_size, CONFIG["seq_length"]),
            "DefaultLSTM (2 layers, emb=128, lstm=256+128)"   : build_model(vocab_size, CONFIG["seq_length"]),
            "DeepLSTM    (3 layers, emb=256, lstm=512+256+128)": build_deep_model(vocab_size, CONFIG["seq_length"]),
        }

        compare_architectures(
            models_dict = arch_models,
            X           = X,
            y           = y,
            epochs      = CONFIG["bonus_epochs"],
            batch_size  = CONFIG["batch_size"],
            n_samples   = CONFIG["bonus_samples"],
        )

    print("\n" + "="*65)
    print("   PIPELINE COMPLETE")
    print(f"  Model saved   : {CONFIG['checkpoint_path']}")
    print(f"  Samples saved : {CONFIG['samples_path']}")
    print(f"  History plot  : {CONFIG['history_plot']}")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
