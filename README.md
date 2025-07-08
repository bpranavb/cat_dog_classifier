# cat_dog_classifier
The goal of this project is to create an image classification model capable of distinguishing between images of cats and dogs. It leverages a Convolutional Neural Network Architechture. The dataset used consists of a large collection of labeled cat and dog images for training, and an unlabeled set for testing predictions.


Google Drive file structure:

MyDrive/
└── classification-data/
    ├── train.zip
    └── test.zip

Cell 1: Mount Google Drive

This cell will prompt you to authorize Google Colab to access your Google Drive. Follow the instructions to click the link, sign in, and paste the authorization code. This is crucial for accessing your train.zip and test.zip files.

Cell 2: Data Loading and Unzipping

This cell defines the paths to your zipped data in Google Drive and creates local directories (/content/extracted_data/train, /content/extracted_data/test) within the Colab environment. It then unzips the train.zip and test.zip files into these respective directories.

Cell 3: Restructure Data into 'cats' and 'dogs' Subfolders

This cell moves the extracted images into cats and dogs subfolders within their respective train and test directories. This structure is required by ImageDataGenerator for automatic labeling.

Note: The test folder images are typically unlabeled (e.g., 1.jpg, 2.jpg), so the restructuring for the test set will result in empty cats and dogs subfolders, which is expected behavior for an unlabeled prediction set.

Cell 4: CNN Model Definition, Training, Evaluation, and Prediction

Data Generators: Sets up ImageDataGenerator instances for training, validation (split from training data), and unlabeled test predictions.

Model Building: Defines the CNN architecture.

Compilation: Compiles the model with binary_crossentropy loss and adam optimizer.

Training: Trains the model for a specified number of epochs.

Evaluation & Saving: Evaluates the model on the validation set and saves the trained model as dog_vs_cat_classifier.h5.

Visualizations: Displays plots for accuracy/loss and a confusion matrix.

Interactive Prediction Loop (Last Section):

After training, this section allows you to enter image filenames (e.g., cat.901.jpg from your train set or 894.jpg from your test set) to get real-time predictions from the trained model. Type quit to exit.

Model Performance
The model's performance is evaluated on a validation set (20% of the original training data). Typical results after 10 epochs are:

Validation Loss: Approximately 0.83

Validation Accuracy: Approximately 0.83

Classification Report (Validation Set):

              precision    recall  f1-score   support

        cats       0.83      0.83      0.83      2500
        dogs       0.83      0.83      0.83      2500

    accuracy                           0.83      5000
   macro avg       0.83      0.83      0.83      5000
weighted avg       0.83      0.83      0.83      5000

New File Structure

/content/
└── extracted_data/
    ├── train/
    │   └── train/      # This is the actual directory containing images after unzipping
    │       ├── cats/   # Labeled cat images moved here
    │       └── dogs/   # Labeled dog images moved here
    └── test/
        └── test/       # This is the actual directory containing images after unzipping
            ├── cats/   # Will be empty as test images are unlabeled
            └── dogs/   # Will be empty as test images are unlabeled
dog_vs_cat_classifier.h5  # The trained model saved here
