# Image Classification using SVM - Fully Explained Tutorial

This tutorial provides a comprehensive guide on image classification using Support Vector Machines (SVM) with Python's `scikit-learn` library. It also delves into K-Nearest Neighbors (KNN) and Decision Trees, allowing you to compare these machine learning techniques for image classification.

**Prerequisites**
  - Download all the files in the repository and put it any folder(eg. Named Image classification)
* **Optional: Anaconda:** While not mandatory, Anaconda offers a convenient environment management system for Python. Download it from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).

- Open Anaconda Navigator
- Create a new environment by going to "Environments tab".
- Install jupyter notebook and Launch.
- Open Image Classification ipynb file which was downloaded earlier.
- Add a cell above the code and use following commands one by one:
- 1. !pip install tensorflow
  2. !pip install matplotlib
  3. !pip install scikit-learn
 
     
**Installation through Anaconda Prompt**

1.  **Create a Virtual Environment (Recommended):**

    - Using Anaconda Prompt or your terminal, create and activate a virtual environment named `image_classification`:

      ```bash
      conda create -n image_classification python=3.x  # Replace 3.x with your desired Python version
      source activate image_classification  # Linux/macOS
      conda activate image_classification        # Windows
      ```

2.  **Install Required Libraries:**

    - in Anaconda prompt write 

      ```bash
      1. pip install tensorflow
  2. pip install matplotlib
  3. pip install scikit-learn
      ```


2.  **Run the Jupyter Notebook:**
    -After installation of libraries , double-click to run the code.

**Explanation of the Code**

The code provided in the `Image_Classification.ipynb` notebook covers the following steps:

  - **Data Loading:** Loads the image dataset from a specified location and preprocesses the images (e.g., resizing, normalization) for better model performance.
  - **Feature Extraction:** Explores different feature extraction techniques, such as converting pixel values to vectors or leveraging pre-trained models like VGG16 or ResNet50 for deeper feature representations.
  - **Model Training:** Trains SVM, KNN, and Decision Tree models on the extracted features and corresponding image labels.
  - **Model Evaluation:** Evaluates the performance of each model using relevant metrics like accuracy, precision, recall, and F1-score.
  - **Visualization (Optional):** Plots confusion matrices, ROC curves, or other visualizations to gain insights into model behavior (encourage customization in the notebook).

**Going Further with KNN and Decision Trees**

While the provided code focuses on SVM, the Jupyter Notebook can be extended to include implementations for KNN and Decision Trees:

**K-Nearest Neighbors (KNN):**

1.  Import the `KNeighborsClassifier` from `sklearn.neighbors`.
2.  Create a KNN classifier instance with desired parameters (e.g., `knn = KNeighborsClassifier(n_neighbors=5)`).
3.  Train the KNN model on the features and labels.
4.  Evaluate the KNN model's performance.

**Decision Trees (DT):**

1.  Import the `DecisionTreeClassifier` from `sklearn.tree`.
2.  Create a Decision Tree classifier instance with desired parameters (e.g., `dt = DecisionTreeClassifier(max_depth=5)`).
3.  Train the Decision Tree model on the features and labels.
4.  Evaluate the Decision Tree model's performance.

**Additional Considerations**

* **Hyperparameter Tuning:** Experiment with different hyperparameters for each model (e.g., C for SVM, n_neighbors for KNN, max_depth for DT) to fine-tune performance for your specific dataset.
* **Feature Engineering:** Explore various feature extraction techniques or dimensionality reduction methods (e.g., PCA) to potentially improve classification accuracy.
