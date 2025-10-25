import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

def load_data():
    """Charge et prépare les données MNIST"""
    print("\n Chargement des données MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"✓ Train: {x_train.shape}, Test: {x_test.shape}")
    return x_train, y_train, x_test, y_test


# ============================================================================
# 2. MODÈLE CNN SIMPLE
# ============================================================================

def build_cnn():

    model = keras.Sequential([

        layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        
        # Bloc 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloc 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    print("## modél CNN")
    model.summary()
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# 3. MODÈLE TRANSFORMER SIMPLE
# ============================================================================

def build_transformer():

    inputs = layers.Input(shape=(28, 28))
    

    x = layers.Dense(64)(inputs)
    
    
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    
    # Feed-forward
    ff = layers.Dense(128, activation='relu')(x)
    ff = layers.Dense(64)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    
    # Classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    print("## modél transformer")
    model.summary()   

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# 4. ENTRAÎNEMENT
# ============================================================================

def train_model(model, x_train, y_train, x_test, y_test, model_name):
    """Entraîne un modèle et retourne les résultats"""
    print(f"\n Entraînement du modèle {model_name}...")
    print(f"Paramètres: {model.count_params():,}")
    
    start_time = time.time()
    
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Évaluation 
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\n {model_name} terminé:")
    print(f"  - Test Accuracy: {test_acc*100:.2f}%")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Temps d'entraînement: {training_time:.1f}s")
    
    return {
        'history': history.history,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'time': training_time,
        'params': model.count_params()
    }


# ============================================================================
# 5. VISUALISATION
# ============================================================================

def plot_comparison(cnn_results, trans_results):
    """Crée des graphiques de comparaison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy pendant l'entraînement
    ax = axes[0, 0]
    ax.plot(cnn_results['history']['accuracy'], 'b-', label='CNN Train', linewidth=2)
    ax.plot(cnn_results['history']['val_accuracy'], 'b--', label='CNN Val', linewidth=2)
    ax.plot(trans_results['history']['accuracy'], 'r-', label='Transformer Train', linewidth=2)
    ax.plot(trans_results['history']['val_accuracy'], 'r--', label='Transformer Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy pendant l\'entraînement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss pendant l'entraînement
    ax = axes[0, 1]
    ax.plot(cnn_results['history']['loss'], 'b-', label='CNN Train', linewidth=2)
    ax.plot(cnn_results['history']['val_loss'], 'b--', label='CNN Val', linewidth=2)
    ax.plot(trans_results['history']['loss'], 'r-', label='Transformer Train', linewidth=2)
    ax.plot(trans_results['history']['val_loss'], 'r--', label='Transformer Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss pendant l\'entraînement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Comparaison Test Accuracy
    ax = axes[1, 0]
    models = ['CNN', 'Transformer']
    accuracies = [cnn_results['test_acc']*100, trans_results['test_acc']*100]
    bars = ax.bar(models, accuracies, color=['#3498db', '#3498db'], alpha=0.7)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Comparaison Test Accuracy')
    min_acc = min(accuracies) - 2
    ax.set_ylim([min_acc, 100])
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Comparaison Temps
    ax = axes[1, 1]
    times = [cnn_results['time'], trans_results['time']]
    bars = ax.bar(models, times, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('Temps (secondes)')
    ax.set_title('Temps d\'entraînement')
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Graphique sauvegardé: comparison.png")
    plt.show()


def print_summary(cnn_results, trans_results):
    """Affiche un tableau récapitulatif"""
    print("\n" + "="*60)
    print("📊 RÉSUMÉ COMPARATIF")
    print("="*60)
    print(f"{'Métrique':<25} {'CNN':<15} {'Transformer':<15}")
    print("-"*60)
    print(f"{'Test Accuracy':<25} {cnn_results['test_acc']*100:<15.2f} {trans_results['test_acc']*100:<15.2f}")
    print(f"{'Test Loss':<25} {cnn_results['test_loss']:<15.4f} {trans_results['test_loss']:<15.4f}")
    print(f"{'Training Time (s)':<25} {cnn_results['time']:<15.1f} {trans_results['time']:<15.1f}")
    print(f"{'Paramètres':<25} {cnn_results['params']:<15,} {trans_results['params']:<15,}")
    print("="*60)
    
    # Meilleur modèle
    if cnn_results['test_acc'] > trans_results['test_acc']:
        print(f"\n🏆 Meilleur modèle: CNN (+{(cnn_results['test_acc']-trans_results['test_acc'])*100:.2f}% accuracy)")
    else:
        print(f"\n🏆 Meilleur modèle: Transformer (+{(trans_results['test_acc']-cnn_results['test_acc'])*100:.2f}% accuracy)")


# ============================================================================
# 6. MAIN - EXÉCUTION
# ============================================================================

def main():
    """Fonction principale"""
    print("="*60)
    print(" ÉTUDE COMPARATIVE: CNN vs TRANSFORMER sur MNIST")
    print("="*60)
    
    # Charger les données
    x_train, y_train, x_test, y_test = load_data()
    
    # Entraîner CNN
    cnn = build_cnn()
    cnn_results = train_model(cnn, x_train, y_train, x_test, y_test, "CNN")
    
    # Entraîner Transformer
    transformer = build_transformer()
    trans_results = train_model(transformer, x_train, y_train, x_test, y_test, "Transformer")
    
    # Afficher les résultats
    print_summary(cnn_results, trans_results)
    
    # Créer les graphiques
    plot_comparison(cnn_results, trans_results)
    
    print("\n Comparaison terminée!")


if __name__ == "__main__":
    main()