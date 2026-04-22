
import os
import re
import numpy as np
import tensorflow as tf

def sample_with_temperature(predictions: np.ndarray,
                             temperature: float = 1.0) -> int:
 
    predictions = np.asarray(predictions).astype("float64")

    
    predictions = np.log(predictions + 1e-8) / temperature

    
    exp_preds   = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    
    probas = np.random.multinomial(1, predictions, 1)
    return int(np.argmax(probas))




def preprocess_seed(seed_text: str) -> list:
   
    cleaned = re.sub(r"[^a-z0-9\s]", "", seed_text.lower())
    return cleaned.split()




def encode_seed(seed_words: list,
                word2idx  : dict,
                seq_length: int) -> list:
  
    sequence = [word2idx.get(w, 0) for w in seed_words]

    if len(sequence) < seq_length:
        
        sequence = [0] * (seq_length - len(sequence)) + sequence
    else:
       
        sequence = sequence[-seq_length:]

    return sequence




def generate_text(model      : tf.keras.Model,
                  seed_text  : str,
                  word2idx   : dict,
                  idx2word   : dict,
                  seq_length : int,
                  num_words  : int   = 100,
                  temperature: float = 0.8) -> str:
    
    
    seed_words = preprocess_seed(seed_text)
    sequence   = encode_seed(seed_words, word2idx, seq_length)

    generated  = list(seed_words)  

    
    for _ in range(num_words):
       
        x = np.array(sequence).reshape(1, seq_length)

       
        preds     = model.predict(x, verbose=0)[0]   

       
        next_idx  = sample_with_temperature(preds, temperature)

        
        next_word = idx2word.get(next_idx, "<UNK>")

        
        generated.append(next_word)

       
        sequence = sequence[1:] + [next_idx]

    return " ".join(generated)




def generate_all_samples(model      : tf.keras.Model,
                         word2idx   : dict,
                         idx2word   : dict,
                         seq_length : int,
                         seeds      : list = None,
                         temperatures: list = None,
                         num_words  : int = 80,
                         save_path  : str = "outputs/generated_samples.txt"):
    
   
    if seeds is None:
        seeds = [
            "to be or not to be",
            "shall i compare thee to",
            "all the worlds a stage",
            "what light through yonder window",
            "friends romans countrymen lend me",
        ]
    if temperatures is None:
        temperatures = [0.5, 0.8, 1.2]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_lines = []

    header = "=" * 70 + "\n  LSTM TEXT GENERATION SAMPLES\n" + "=" * 70
    print(header)
    output_lines.append(header)

    for seed in seeds:
        section = f"\n{'─'*70}\n  SEED : \"{seed}\"\n"
        print(section)
        output_lines.append(section)

        for temp in temperatures:
            generated = generate_text(
                model       = model,
                seed_text   = seed,
                word2idx    = word2idx,
                idx2word    = idx2word,
                seq_length  = seq_length,
                num_words   = num_words,
                temperature = temp
            )
            block = f"\n  [Temperature = {temp}]\n  {generated}\n"
            print(block)
            output_lines.append(block)

   
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[Generate] All samples saved to: {save_path}")
