# Detection of Eccentric Viewing in Low Vision Patients

## Overview
This project focuses on detecting **eccentric viewing behavior** in low vision patients using deep learning techniques. Eccentric viewing is a compensatory strategy used by individuals with central vision loss, where they rely on a peripheral retinal region called the Preferred Retinal Locus (PRL) instead of the damaged fovea.
The system analyzes eye movement video data to predict gaze coordinates and identify eccentric viewing patterns, helping support vision rehabilitation and remote monitoring applications.


## Problem Statement
Traditional methods for detecting eccentric viewing rely on specialized eye-tracking hardware and clinical supervision, which are expensive, time-consuming, and not easily accessible. This project aims to provide a **cost-effective, automated, and scalable AI-based solution** for eccentric viewing detection.


## Features
- Eye movement video processing  
- Frame extraction and preprocessing  
- Spatial feature extraction using CNN  
- Temporal sequence modeling using LSTM  
- Gaze coordinate prediction  
- Eccentric viewing detection  
- Hyperparameter tuning for improved accuracy  


## Dataset
This project uses the **EYEDIAP Dataset**, a benchmark dataset for gaze estimation and eye movement analysis.

Dataset includes:
- Eye movement videos  
- Gaze labels (x, y coordinates)  
- Head pose information  



## Tech Stack
- Python
- TensorFlow  
- Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  
- Convolutional Neural Network (CNN)  
- Long Short-Term Memory (LSTM)  



## Workflow
1. Video Input  
2. Frame Extraction  
3. Data Preprocessing  
4. CNN Feature Extraction  
5. LSTM Sequence Modeling  
6. Gaze Prediction  
7. Eccentric Viewing Detection  

## Results
The hybrid CNN + LSTM model significantly improves gaze prediction accuracy by combining spatial and temporal learning. Hyperparameter tuning further reduces prediction error, making the system more reliable for real-world applications.


## Applications
- Vision rehabilitation  
- Low vision patient monitoring  
- Remote healthcare systems  
- Assistive technologies  

## Future Work
- Real-time gaze tracking  
- Mobile deployment  
- Transformer-based architectures  
- Larger datasets for better generalization  


## Author
Ahir Jaimi , Twisha Kamani  
B.Tech Artificial Intelligence and Machine Learning  

## License
This project is for academic and research purposes.
