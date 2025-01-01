# Gesture Recognition Project

## Overview
This project focuses on designing and implementing a system to recognize five predefined hand gestures to control a smart television. The gestures are captured as video sequences through a webcam, processed, and classified into specific commands. This system eliminates the need for a remote control, offering a hands-free and user-friendly interaction method.

### Supported Gestures:
1. **Thumbs Up**: Increase the volume.
2. **Thumbs Down**: Decrease the volume.
3. **Left Swipe**: Jump backward 10 seconds.
4. **Right Swipe**: Jump forward 10 seconds.
5. **Stop**: Pause the video.

## Objectives
- Develop a robust gesture recognition model capable of high accuracy and generalization.
- Ensure real-time processing capabilities with minimal latency.
- Optimize the architecture for a balance between performance and efficiency.
- Iteratively improve the system through multiple experiments and refinements.

## Project Structure
- **Data Preprocessing**: Includes video frame resizing, normalization, and advanced augmentation techniques.
- **Model Architectures**:
  - **Conv2D + GRU**: Processes spatial and temporal information using 2D convolutional layers and GRUs.
  - **Conv3D**: Extracts spatiotemporal patterns using 3D convolutions.
  - **Fusion Model**: Combines outputs from Conv2D + GRU and Conv3D for enhanced performance.
- **Experiments**: Iteratively improve the models by testing different augmentations, architectures, and optimization strategies.

## Installation
### Prerequisites
- Python 3.8 or above
- TensorFlow 2.5 or above
- OpenCV
- NumPy
- Pandas
- Matplotlib

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gesture-recognition.git
   ```
2. Navigate to the project directory:
   ```bash
   cd gesture-recognition
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Data Preparation
1. Place your training and validation video sequences in the respective folders:
   - `Project_data/train`
   - `Project_data/val`
2. Ensure the CSV files `train.csv` and `val.csv` are correctly formatted with columns for `folder_name`, `gesture_name`, and `label`.
3. Preprocess and augment the data using the provided scripts.

### Training
Run the training script:
```bash
python train.py
```
This script trains the models defined in the experiments and saves the best-performing models to the disk.

### Evaluation
To evaluate the trained models on validation data, use the evaluation script:
```bash
python evaluate.py
```

### Visualization
To visualize augmented data or model predictions, run:
```bash
python visualize.py
```

## Experiments
### Experiment Summary
- **Experiment 1**: Baseline models with no data augmentation.
- **Experiment 2**: Added augmentation and enhanced GRU layers.
- **Experiment 3**: Simplified architectures with class balancing.
- **Experiment 4**: Refined augmentations and added gradient clipping.
- **Experiment 5**: Fusion model integrating Conv2D + GRU and Conv3D outputs.

### Best Model
The **Fusion Model** from Experiment 5 achieved the best performance with:
- **Validation Accuracy**: Highest among all experiments.
- **Robustness**: Improved generalization due to advanced augmentation and class balancing.

## File Structure
```
gesture-recognition/
|-- Project_data/
|   |-- train/
|   |-- val/
|   |-- train.csv
|   |-- val.csv
|-- models/
|   |-- conv2d_gru_model.h5
|   |-- conv3d_model.h5
|   |-- fusion_model.h5
|-- scripts/
|   |-- train.py
|   |-- evaluate.py
|   |-- visualize.py
|-- README.md
|-- requirements.txt
```

## Future Work
- Incorporate attention mechanisms for improved temporal modeling.
- Test on larger, real-world datasets for better validation.
- Optimize the fusion model for deployment on edge devices.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
Special thanks to all contributors and resources used during this project.
