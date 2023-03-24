# Supervised Universal Transformations
Learning transformations between vision datasets to overcome sensor hardware versioning

## Overview
The main idea of this project is to come up with a universal transformation between dataset distributions.
The difference between these dataset distributions are a result of hardware versioning between sensors.
This transformation should preserve several properties of the original dataset:
1. The prediction of the base model.
2. The structure and features present in the image.


![camera versioning](figures/camera-versioning.jpg)

## Training

```python
pip install -r requirements.txt
cd training
python train_lenscoder.py <path-to-dataset>
```