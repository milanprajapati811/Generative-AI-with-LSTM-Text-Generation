
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout



def build_model(vocab_size: int,
                seq_length: int,
                embedding_dim: int  = 128,
                lstm_units_1: int   = 256,
                lstm_units_2: int   = 128,
                dropout_rate: float = 0.3,
                learning_rate: float = 1e-3) -> tf.keras.Model:
   
    model = Sequential([

        
        Embedding(
            input_dim    = vocab_size,
            output_dim   = embedding_dim,
            input_length = seq_length,
            name         = "embedding"
        ),

        LSTM(
            units            = lstm_units_1,
            return_sequences = True,
            name             = "lstm_1"
        ),
        Dropout(dropout_rate, name="dropout_1"),

      
        LSTM(
            units            = lstm_units_2,
            return_sequences = False,
            name             = "lstm_2"
        ),
        Dropout(dropout_rate, name="dropout_2"),

      
        Dense(vocab_size, activation="softmax", name="output"),

    ], name="DefaultLSTM")

    
    model.compile(
        loss      = "sparse_categorical_crossentropy",
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics   = ["accuracy"]
    )

    model.summary()
    return model




def build_shallow_model(vocab_size: int,
                        seq_length: int) -> tf.keras.Model:
   
    model = Sequential([
        Embedding(vocab_size, 64, input_length=seq_length, name="embedding"),
        LSTM(128, name="lstm_1"),
        Dropout(0.3, name="dropout_1"),
        Dense(vocab_size, activation="softmax", name="output"),
    ], name="ShallowLSTM")

    model.compile(
        loss      = "sparse_categorical_crossentropy",
        optimizer = "adam",
        metrics   = ["accuracy"]
    )
    return model




def build_deep_model(vocab_size: int,
                     seq_length: int) -> tf.keras.Model:
    
    model = Sequential([
        Embedding(vocab_size, 256, input_length=seq_length, name="embedding"),
        LSTM(512, return_sequences=True, name="lstm_1"),
        Dropout(0.4, name="dropout_1"),
        LSTM(256, return_sequences=True, name="lstm_2"),
        Dropout(0.4, name="dropout_2"),
        LSTM(128, return_sequences=False, name="lstm_3"),
        Dropout(0.3, name="dropout_3"),
        Dense(vocab_size, activation="softmax", name="output"),
    ], name="DeepLSTM")

    model.compile(
        loss      = "sparse_categorical_crossentropy",
        optimizer = "adam",
        metrics   = ["accuracy"]
    )
    return model



def compare_architectures(vocab_size: int, seq_length: int) -> None:
   
    models = {
        "ShallowLSTM": build_shallow_model(vocab_size, seq_length),
        "DefaultLSTM": build_model(vocab_size, seq_length),
        "DeepLSTM"   : build_deep_model(vocab_size, seq_length),
    }

    print("\n" + "="*55)
    print("  ARCHITECTURE PARAMETER COMPARISON")
    print("="*55)
    print(f"{'Model':<20} {'Total Params':>15} {'Trainable':>15}")
    print("-"*55)
    for name, m in models.items():
        total     = m.count_params()
        trainable = sum(tf.keras.backend.count_params(w)
                        for w in m.trainable_weights)
        print(f"{name:<20} {total:>15,} {trainable:>15,}")
    print("="*55)
