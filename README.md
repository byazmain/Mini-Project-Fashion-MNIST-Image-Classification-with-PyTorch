# Mini-Project-Fashion-MNIST-Image-Classification-with-PyTorch
# Fashion-MNIST Image Classification with PyTorch (ANN)


A fully connected neural network (ANN) built with PyTorch to classify clothing images from the Fashion-MNIST dataset. The project covers the full workflow: data loading, normalization, a custom `Dataset` class, model building, a manual training loop on GPU, and evaluation.

## Results

| Metric | Train | Test |
|---|---|---|
| Accuracy (epoch 15) | **91.57%** | **88.94%** |
| Loss (epoch 15) | 0.2216 | 0.3234 |

- Best test accuracy: **89.01%** (epoch 11)
- Lowest test loss: **0.2975** (epoch 12)

<!-- Save the plot from the last notebook cell as images/loss_curve.png and the epoch log as images/training_log.png, then keep these lines. -->
<img width="779" height="594" alt="image" src="https://github.com/user-attachments/assets/66ff8b8f-1993-4c21-83e7-9c1a200dcc39" />


### Observations

- **Mild overfitting.** The gap between train and test accuracy is about 2.6 percentage points. In an earlier experiment on a 6,000-image subset, the gap was over 10 points, so using the full 60,000 images was the most effective fix.
- **Early stopping would help.** After epoch 12, the test loss stops improving while the train loss keeps falling.
- **Typical range.** 88-90% is a normal result for a plain fully connected network on this dataset. Going clearly above 90% needs a different model type, such as a CNN.

## Dataset

[Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) (Zalando Research, via Kaggle)

- 60,000 training images and 10,000 test images
- 28 x 28 grayscale images, stored as 784 pixel columns plus a `label` column
- 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Model and Training Setup

| Item | Value |
|---|---|
| Architecture | 784 → 256 → 128 → 64 → 10 |
| Activation | ReLU |
| Loss function | CrossEntropyLoss |
| Optimizer | SGD (learning rate 0.1) |
| Batch size | 32 |
| Epochs | 15 |
| Preprocessing | Pixel values divided by 255 |
| Hardware | GPU (CUDA) |

## Project Structure

```
.
├── ann_project_Fminst_bigdata_gpu.ipynb   # Full notebook
├── images/                                # Result plots used in this README
└── README.md
```

The dataset files (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`) are not included. Download them from Kaggle (link above).

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install the requirements:
   ```bash
   pip install torch pandas numpy scikit-learn matplotlib jupyter
   ```
3. Download the dataset from Kaggle and place `fashion-mnist_train.csv` and `fashion-mnist_test.csv` in the project folder.
4. Open `ann_project_Fminst_bigdata_gpu.ipynb` and run all cells. A GPU is recommended (for example, Google Colab with a GPU runtime), but the notebook also runs on CPU.

## Future Improvements

- Add a validation split and early stopping
- Add Dropout, BatchNorm, and weight decay
- Switch to Adam with a learning-rate scheduler
- Build a CNN and compare it with the ANN (expected to reach about 92-93%)
- Add a confusion matrix to study the confusion between Shirt, T-shirt/top, and Pullover

## Author

**Azmain**
CSE student at Jagannath University | Aspiring AI/ML Engineer

[LinkedIn](https://www.linkedin.com/in/<your-profile>) | [GitHub](https://github.com/<your-username>)
