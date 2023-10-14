import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, concatenate
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray

# inital settings...
input_shape_cnn1 = (28, 28, 1)
input_shape_cnn2 = (40, 40, 1)
cnn1_config = {
    'conv1_filters': 32,
    'conv1_kernel': (5, 5),
    'pool1_size': (2, 2),
    'dense_units': 192,
}
cnn2_config = {
    'conv1_filters': 64,
    'conv1_kernel': (11, 11),
    'pool1_size': (2, 2),
    'dense_units': 192,
}
rpn_hyperparameters = {
    'rpn_conv_filters': 256,
    'rpn_conv_kernel': (3, 3),
    'rpn_pool_size': (2, 2),
}

#  CNN1
def create_cnn1(input_shape):
    model = tf.keras.Sequential()
    model.add(Conv2D(cnn1_config['conv1_filters'], cnn1_config['conv1_kernel'], activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=cnn1_config['pool1_size']))
    model.add(Flatten())
    model.add(Dense(cnn1_config['dense_units'], activation='relu'))
    return model

#  CNN2
def create_cnn2(input_shape):
    model = tf.keras.Sequential()
    model.add(Conv2D(cnn2_config['conv1_filters'], cnn2_config['conv1_kernel'], activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=cnn2_config['pool1_size']))
    model.add(Flatten())
    model.add(Dense(cnn2_config['dense_units'], activation='relu'))
    return model

#  RPN
def create_rpn(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(rpn_hyperparameters['rpn_conv_filters'], rpn_hyperparameters['rpn_conv_kernel'], activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=rpn_hyperparameters['rpn_pool_size'])(x)
    x = Flatten()(x)
    rpn_model = Model(inputs=input_layer, outputs=x)
    return rpn_model

# Load  MRI images
def load_and_preprocess_images(image_paths, input_shape):
    images = [resize(rgb2gray(io.imread(path)), input_shape[:2], anti_aliasing=True) for path in image_paths]
    images = np.array(images).reshape(-1, *input_shape)
    return images

# Create-compile  models
cnn1_model = create_cnn1(input_shape_cnn1)
cnn2_model = create_cnn2(input_shape_cnn2)
rpn_model = create_rpn(input_shape_cnn1)  

# Loading  MRI 
image_paths = ["brat\s1.jpg", "hw\s2.jpg"]  
MRI_images_cnn1 = load_and preprocces_images(image_paths, input_shape_cnn1)
MRI_images_cnn2 = load_and_preprocess_images(image_paths, input_shape_cnn2)

# Extracting features 
features_cnn1 = cnn1_model.predict(MRI_images_cnn1)
features_cnn2 = cnn2_model.predict(MRI_images_cnn2)


rpn_features = rpn_model.predict(MRI_images_cnn1)  


super_feature_vector = np.concatenate((features_cnn1, features_cnn2, rpn_features), axis=1)


labels = [0, 1]  # 0 benign , 1 malignant


rf_classifier = RandomForestClassifier(n_estimators=100)
svm_classifier = SVC(kernel='linear')

rf_classifier.fit(super_feature_vector, labels)
svm_classifier.fit(super_feature_vector, labels)


new_MRI_images_cnn1 = load_and_preprocess_images(["new_image_path1.jpg"], input_shape_cnn1)
new_MRI_images_cnn2 = load_and_preprocess_images(["new_image_path2.jpg"], input_shape_cnn2)

new_features_cnn1 = cnn1_model.predict(new_MRI_images_cnn1)
new_features_cnn2 = cnn2_model.predict(new_MRI_images_cnn2)

new_rpn_features = rpn_model.predict(new_MRI_images_cnn1)  
new_super_feature_vector = np.concatenate((new_features_cnn1, new_features_cnn2, new_rpn_features), axis=1)

rf_predictions = rf_classifier.predict(new_super_feature_vector)
svm_predictions = svm_classifier.predict(new_super_feature_vector)

