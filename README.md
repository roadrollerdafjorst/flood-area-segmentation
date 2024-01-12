# Flood Area Segmentation with Deep Learning

## Introduction

This project focuses on developing a flood area segmentation application using well-established deep learning models such as UNet and DeepLabV3. The application processes satellite and aerial images to identify and highlight flood-affected regions with precision. The achieved DICE scores of 0.82 with UNet and 0.86 with DeepLabV3 showcase the effectiveness of the models in this context.

## Features

- **Semantic Segmentation:** Utilizes widely recognized deep learning models (UNet and DeepLabV3) for precise semantic segmentation of flood-affected regions.
- **Satellite and Aerial Image Support:** Processes both satellite and aerial images for identifying flood-affected areas.
- **High Precision:** Attains competitive DICE scores (0.82 with UNet, 0.86 with DeepLabV3) for accurate flood area identification.

## Installation

To install and run the flood area segmentation application, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/roadrollerdafjorst/flood-area-segmentation.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    streamlit run app.py
    ```

## Usage

1. Launch the application as described in the installation section.

2. Upload the satellite or aerial image you want to process.

3. Choose the segmentation model (UNet or DeepLabV3) and set the required parameters.

4. Click the "Segment" button to start the flood area segmentation process.

5. The segmented image will be displayed, with flood-affected regions highlighted.

6. Save the segmented image or conduct further analysis as needed.

## Results

The deep learning models used in this project have demonstrated commendable results:

- UNet: DICE Score of 0.82
- DeepLabV3: DICE Score of 0.86
