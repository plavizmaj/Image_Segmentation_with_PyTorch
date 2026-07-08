# Human Image Segmentation with PyTorch

This project trains a binary image segmentation model that separates a human subject from the background in a photo. It uses a U-Net architecture built with `segmentation_models_pytorch`, an EfficientNet encoder pretrained on ImageNet, and the Human Segmentation Dataset for training data. The notebook covers the full pipeline: loading and splitting the dataset, applying data augmentation, building a custom PyTorch `Dataset` and `DataLoader`, defining the segmentation model and loss, training with validation based checkpointing, and running inference on a held out image to produce a predicted mask.

It is intended for people who want a working, end to end example of a segmentation training pipeline in PyTorch, whether that is for learning how the pieces fit together (data augmentation, custom datasets, combined Dice and BCE loss, training loop) or as a starting point to adapt to a different segmentation dataset.

## Features

- Loads image and mask paths from a CSV file (`train.csv`) and splits them into training and validation sets (80/20) using `train_test_split`.
- Applies augmentations with the `albumentations` library: resizing to a fixed size, random horizontal flip, and random vertical flip on the training set (validation only gets resized).
- Wraps the data in a custom `SegmentationDataset` class that reads images with OpenCV, converts them to RGB, reads the corresponding mask in grayscale, applies augmentations, and converts both to normalized PyTorch tensors in `(C, H, W)` format.
- Defines a `SegmentationModel` class (a `nn.Module` wrapper around `smp.Unet`) that returns both the raw logits and a combined loss (`DiceLoss` + `BCEWithLogitsLoss`) when masks are provided during training.
- Includes separate `train_fn` and `eval_fn` functions that loop over the data loaders, move batches to the GPU, and report the average loss per epoch.
- Saves the model's weights (`best_model.pt`) automatically whenever validation loss improves.
- Runs inference on a validation image, applies a sigmoid to the logits, and thresholds the output at 0.5 to produce a binary predicted mask.
- Uses a `helper` module (imported separately, not part of this notebook) to visualize the image, ground truth mask, and predicted mask side by side.

## Project Structure

Since this project is a single Jupyter notebook, the "structure" corresponds to the task sections inside it rather than separate files:

- **Task 1, Set up environment and imports**: installs `segmentation-models-pytorch` and `albumentations`, and imports all required libraries (`torch`, `cv2`, `numpy`, `pandas`, `matplotlib`, `sklearn`, `tqdm`, `albumentations`, `segmentation_models_pytorch`).
- **Task 2, Setup Configurations**: defines the paths to the dataset (`CSV_FILE`, `DATA_DIR`), the device (`cuda`), and the training hyperparameters (`EPOCHS`, `LR`, `IMAGE_SIZE`, `BATCH_SIZE`, encoder name, and pretrained weights).
- **Task 3, Augmentation Functions**: `get_train_augs()` and `get_val_augs()`, which build the `albumentations` pipelines described above.
- **Task 4, Create Custom Dataset**: the `SegmentationDataset` class that reads, augments, and formats each image/mask pair.
- **Task 5, Load dataset into batches**: wraps the training and validation datasets in `DataLoader` objects.
- **Task 6, Create Segmentation Model**: the `SegmentationModel` class built on `smp.Unet`.
- **Task 7, Create Train and Validation Function**: `train_fn` and `eval_fn`, the training and evaluation loops.
- **Task 8, Train Model**: the main training loop over `EPOCHS`, with checkpoint saving to `best_model.pt`.
- **Task 9, Inference**: loads the saved weights and runs a prediction on a sample validation image.

Two external resources are expected alongside the notebook:

- `train.csv`, a CSV file with `images` and `masks` columns pointing to the dataset files (from the Human Segmentation Dataset).
- `helper.py`, a local module imported as `import helper` and used for the `helper.show_image(...)` visualization calls. This file is not included in the notebook and needs to be present in the working directory.

## Installation

This project requires an NVIDIA GPU with CUDA support. The configuration cell sets `DEVICE = 'cuda'`, and both the model and every batch of images/masks are moved to that device during training and inference, so a CPU only setup will not work without modifying the code.

1. Make sure you have a CUDA capable GPU and the matching NVIDIA drivers installed.
2. Install a CUDA enabled build of PyTorch (check `https://pytorch.org` for the correct command for your CUDA version), for example:
```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
3. Install the remaining dependencies:
```bash
   pip install segmentation-models-pytorch
   pip install -U git+https://github.com/albumentations-team/albumentations
   pip install --upgrade opencv-contrib-python
   pip install pandas numpy matplotlib scikit-learn tqdm
```
4. Obtain the dataset (the notebook clones it from the original author's repository):
```bash
   git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git
```
5. Place a `helper.py` file with a `show_image(image, mask, pred_mask=None)` function in your working directory, since the notebook imports and calls it directly.

## Usage

1. Update the paths in the configuration cell to match your local setup:
```python
   CSV_FILE = '/path/to/Human-Segmentation-Dataset-master/train.csv'
   DATA_DIR = '/path/to/data/'
```
2. Run the notebook cells in order, from Task 1 through Task 9. Key steps along the way:
   - Build the datasets and loaders:
```python
     trainset = SegmentationDataset(train_df, get_train_augs())
     validset = SegmentationDataset(valid_df, get_val_augs())

     trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
     validloader = DataLoader(validset, batch_size=BATCH_SIZE)
```
   - Create the model and move it to the GPU:
```python
     model = SegmentationModel()
     model.to(DEVICE)
```
   - Train for the configured number of epochs, saving the best checkpoint automatically:
```python
     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

     best_valid_loss = np.inf
     for i in range(EPOCHS):
         train_loss = train_fn(trainloader, model, optimizer)
         valid_loss = eval_fn(validloader, model)

         if valid_loss < best_valid_loss:
             torch.save(model.state_dict(), 'best_model.pt')
             best_valid_loss = valid_loss

         print(f"Epoch: {i+1} Train_loss: {train_loss} Valid_loss: {valid_loss}")
```
   - Load the saved weights and run inference on a validation sample:
```python
     model.load_state_dict(torch.load("best_model.pt"))

     image, mask = validset[idx]
     logits_mask = model(image.to(DEVICE).unsqueeze(0))
     pred_mask = torch.sigmoid(logits_mask)
     pred_mask = (pred_mask > 0.5) * 1.0
```

## Configuration

All configuration is done through plain Python variables at the top of the notebook (Task 2), there are no external config files or environment variables:

| Variable | Purpose | Value in notebook |
|---|---|---|
| `CSV_FILE` | Path to the CSV with image/mask paths | `/content/Human-Segmentation-Dataset-master/train.csv` |
| `DATA_DIR` | Base data directory | `/content/` |
| `DEVICE` | Compute device, requires a GPU | `cuda` |
| `EPOCHS` | Number of training epochs | `25` |
| `LR` | Learning rate for the Adam optimizer | `0.003` |
| `IMAGE_SIZE` | Size (height and width) images are resized to | `320` |
| `BATCH_SIZE` | Batch size for both loaders | `16` |
| `ENCODER` | Backbone encoder for the U-Net | `timm-efficientnet-b0` |
| `WEIGHTS` | Pretrained weights for the encoder | `imagenet` |

Update `CSV_FILE` and `DATA_DIR` to point to wherever you place the cloned dataset on your own machine.

## Technologies Used

- Python
- PyTorch (`torch`, `torch.nn`, `torch.utils.data`)
- `segmentation-models-pytorch` (U-Net architecture and Dice loss)
- `albumentations` (image augmentation)
- OpenCV (`opencv-contrib-python`) for image and mask reading
- `pandas` and `numpy` for data handling
- `scikit-learn` (`train_test_split`) for the train/validation split
- `matplotlib` for visualization
- `tqdm` for progress bars during training and evaluation

## Author

*M. E. Petra Bošković*
