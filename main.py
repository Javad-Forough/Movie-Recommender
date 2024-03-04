from data_loader import load_data, preprocess_data
from text_preprocessor import preprocess_text_data
from model import build_neural_network, train_model
import tensorflow as tf

def main():
    # Load and preprocess data
    data = load_data("TMDb_updated.csv")
    data = preprocess_data(data)
    text_features = preprocess_text_data(data)

    # Prepare numerical features
    numerical_features = data[['vote_count', 'vote_average']].values

    # Concatenate numerical and text features
    all_features = numerical_features
    all_features = tf.keras.layers.concatenate([all_features, text_features.toarray()])

    # Build and train neural network model
    model = build_neural_network(all_features.shape[1])
    target = data['vote_average'].values
    train_model(model, all_features, target)

if __name__ == "__main__":
    main()
