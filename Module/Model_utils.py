"""
Utility functions for saving and loading models with custom layers.
"""
import tensorflow as tf
from typing import Dict, Any


def get_custom_objects() -> Dict[str, Any]:
    """
    Get custom objects dictionary for loading models with Lambda layers.
    
    Returns:
        Dictionary of custom objects
    """
    return {
        'tf': tf,
    }


def save_model_safe(model: tf.keras.Model, filepath: str):
    """
    Save model safely with custom objects.
    
    Args:
        model: Keras model to save
        filepath: Path to save model (.h5 or SavedModel)
    """
    print(f"Saving model to: {filepath}")
    
    try:
        # Try saving as .h5
        if filepath.endswith('.h5'):
            model.save(filepath, save_format='h5')
        else:
            model.save(filepath)
        
        print(f"✅ Model saved successfully")
        
    except Exception as e:
        print(f"⚠️  Standard save failed: {e}")
        print("Trying to save weights only...")
        
        # Fallback: save weights
        weights_path = filepath.replace('.h5', '_weights.h5')
        model.save_weights(weights_path)
        print(f"✅ Model weights saved to: {weights_path}")


def load_model_safe(filepath: str, custom_objects: Dict[str, Any] = None) -> tf.keras.Model:
    """
    Load model safely with custom objects.
    
    Args:
        filepath: Path to model file
        custom_objects: Custom objects dictionary (optional)
        
    Returns:
        Loaded Keras model
    """
    if custom_objects is None:
        custom_objects = get_custom_objects()
    
    print(f"Loading model from: {filepath}")
    
    try:
        # Try loading full model
        model = tf.keras.models.load_model(
            filepath,
            compile=False,
            custom_objects=custom_objects
        )
        print(f"✅ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"⚠️  Failed to load full model: {e}")
        print("This is a known issue with complex Lambda layers.")
        print("\nPlease use one of these solutions:")
        print("1. Rebuild model and load weights:")
        print("   from Module.Model_isolated_sign import build_phase1_model")
        print("   model = build_phase1_model()")
        print("   model.load_weights('path/to/weights.h5')")
        print("\n2. Or save model in SavedModel format (folder) instead of .h5")
        raise


def load_model_with_architecture(
    weights_path: str,
    architecture_builder,
    **model_kwargs
) -> tf.keras.Model:
    """
    Load model by rebuilding architecture and loading weights.
    
    This is the most reliable method for models with complex layers.
    
    Args:
        weights_path: Path to model weights (.h5)
        architecture_builder: Function that builds model architecture
        **model_kwargs: Keyword arguments for architecture builder
        
    Returns:
        Model with loaded weights
        
    Example:
        from Module.Model_isolated_sign import build_phase1_model
        
        model = load_model_with_architecture(
            'models/phase1/final_model.h5',
            build_phase1_model,
            num_keypoints=141,
            num_classes=30,
            max_frames=180
        )
    """
    print(f"Building model architecture...")
    model = architecture_builder(**model_kwargs)
    
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path)
    
    print(f"✅ Model loaded successfully with custom architecture")
    return model


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("MODEL UTILITIES - TESTING")
    print("=" * 60)
    
    # Test building and saving model
    from Module.Model_isolated_sign import build_phase1_model
    
    print("\n1. Building test model...")
    model = build_phase1_model(
        num_keypoints=141,
        num_classes=30,
        max_frames=180
    )
    
    print("\n2. Testing save...")
    test_save_path = "test_model.h5"
    save_model_safe(model, test_save_path)
    
    print("\n3. Testing load with custom objects...")
    try:
        loaded_model = load_model_safe(test_save_path)
        print("✅ Load successful!")
    except Exception as e:
        print(f"⚠️  Load failed (expected): {e}")
        
        print("\n4. Testing load with architecture rebuild...")
        loaded_model = load_model_with_architecture(
            test_save_path,
            build_phase1_model,
            num_keypoints=141,
            num_classes=30,
            max_frames=180
        )
        print("✅ Load with rebuild successful!")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    
    # Cleanup
    import os
    if os.path.exists(test_save_path):
        os.remove(test_save_path)
        print("\n✓ Test file cleaned up")
