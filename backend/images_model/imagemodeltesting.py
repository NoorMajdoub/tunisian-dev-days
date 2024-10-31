from tensorflow.keras.models import load_model

# Load the best model from the checkpoint
ensemble_model = load_model('best_weighted_ensemble_model.keras')
from tensorflow.keras.preprocessing import image
import numpy as np

"""
img_path = 'testingimageforresnet.png'  # Replace with actual image path

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1] range
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
prediction = ensemble_model.predict(img_array)
predicted_class = int(prediction[0] > 0.5)
print(f'Predicted class: {predicted_class}')#gives 0 0 is normla , for false , we good
"""
#pip install pydot
from tensorflow.keras.utils import plot_model

# Plot the model architecture and save to a file
plot_model(ensemble_model, to_file='ensemble_model_architecture.png', show_shapes=True, show_layer_names=True)
"""
for layer in ensemble_model.layers:
    print(layer.name)
"""