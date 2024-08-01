from unet_vgg19 import build_model
from keras.utils import plot_model

model = build_model()
model.summary()
plot_model(model, show_shapes=True, show_layer_activations=True)
