# Dynamic Model Configuration Complete

I have successfully updated the app to support your 12 base models and the stacking ensembles. The user can now dynamically choose the model architecture, the number of channels, and the number of classes, directly from the web interface.

## Changes Made

### 1. Frontend Updates (`index.html`)
- **Configuration Bar**: Added three dropdowns in the hero section below the title to allow the selection of:
  - **Model**: BiLSTM, CNN, Transformer, or Stacking Ensemble.
  - **Channels**: 5 Channels or 2 Channels.
  - **Classes**: 3 Classes (W, NR, R) or 5 Classes (W, N1, N2, N3, R).
- **Dynamic Interface**: When 2 channels are selected, the EOG and EMG signal visualizations in the "Aperçu des canaux" and the simulation panel are automatically hidden, leaving only the two EEG channels.
- **Dynamic Charting**: The rendering logic for the hypnogram, the Stage Breakdown, and the Time Distribution Bar Chart now dynamically adapts to whether 3 or 5 classes are selected. It uses the `class_names` returned by the server (e.g. adding N1, N2, and N3 with their respective colors).

### 2. Backend Updates (`sleep_server.py`)
- **Added Model Architectures**: Integrated the definitions for `SleepCNN` and `SleepTransformer` (including the `PositionalEncoding` module) alongside the existing `SleepLSTM`.
- **Dynamic Configuration**: Modified the model classes to accept `n_channels` and `num_classes` dynamically upon initialization.
- **Model Caching (`_MODEL_CACHE`)**: The backend now features a caching mechanism. Instead of loading models during server startup, models are loaded the first time a specific configuration is requested, and then cached in memory for subsequent requests.
- **Stacking Inference**: Added `get_stacking_ensemble` which loads the `joblib` meta-learner and the 3 base models (CNN, LSTM, Transformer) for the requested configuration. It extracts the probabilities (`softmax`), concatenates them, and feeds them into the meta-learner to get the final prediction.
- **AASM Metrics**: `compute_aasm_stats` was upgraded to correctly group `NREM` statistics (summing N1, N2, N3 if 5 classes are chosen) so that the clinical alerts still function properly.

## How to Test
1. Make sure your Python environment has `joblib` installed (`pip install joblib`), as it's required for the stacking ensemble `.pkl` files.
2. Run the server: `python sleep_server.py`.
3. Open `index.html` in your browser.
4. Try uploading an `.edf` file with different configurations (e.g., CNN with 2 channels and 3 classes, or Stacking with 5 channels and 5 classes) and check the resulting hypnograms!

> [!TIP]
> The stacking ensemble requires loading 3 PyTorch models simultaneously. The first time you select a new configuration, it may take a few seconds to load the weights into memory. Subsequent requests with the same configuration will be very fast due to caching.
