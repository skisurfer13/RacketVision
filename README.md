

# 🎾 RacketVision: Computer Vision-Based Tennis Analysis System

**RacketVision** is an advanced sports analytics tool utilizing state-of-the-art computer vision architectures to provide detailed insights into player and ball movements. The system integrates **TrackNet**, **YOLOv8**, **CNN**, and **CatBoost Regressor**, delivering accurate player detection, ball tracking, and speed estimation. 

## 📹 Demo Video
[![Watch the demo](https://img.youtube.com/vi/your-demo-video-id/maxresdefault.jpg)](https://youtu.be/1os459edW1c?si=70TN8UUiDFRaZyHy)

> Click on the thumbnail above to watch a demonstration of **RacketVision** in action, showcasing real-time analysis, player and ball tracking, and metric visualization.

## 🌟 Key Features

- **TrackNet for Tennis Ball Tracking**: Provides precise, real-time tracking of tennis ball movement during rallies.
- **YOLOv8 for Player Detection**: Accurately tracks player movement across the court, even under complex motion scenarios.
- **CNN for Court Feature Extraction**: Identifies key court features, ensuring accurate mapping of player and ball positions relative to the court boundaries.
- **CatBoost Regressor for Ball Bounce Prediction**: Predicts the bounce location of the tennis ball with high accuracy, supporting the Hawk-Eye decision system.
- **Mini-Map Visualization**: Real-time mini-map displaying the positions and movements of players and the ball, offering intuitive game dynamics visualization.
- **Comprehensive Metrics**: Computes player speed, shot speed, shot depth, and provides average performance statistics for players.
- **Hawk-Eye System Integration**: Automatically detects and highlights out-of-bound shots using court boundary mapping.

## 🔧 Architecture Overview

### 1. **TrackNet for Ball Tracking**
   - **Deep Learning Model**: TrackNet leverages a VGG-16-based architecture to track high-speed tennis balls in standard broadcast videos using a heatmap-based approach for motion detection.
   - **Multi-Frame Input**: Processes consecutive frames for improved accuracy, particularly when the ball is blurry or occluded.

### 2. **YOLOv8 for Player Detection**
   - **Object Detection Algorithm**: YOLOv8 ensures fast and robust player detection, even under dynamic movements and occlusions, facilitating reliable player tracking.

### 3. **CNN for Court Feature Extraction**
   - **Feature Detection**: Extracts key court features, such as lines and net, allowing the system to accurately map player and ball positions.
   - **Homography Transformation**: Ensures precise mapping of predicted court points to reference court points, improving accuracy in occlusion scenarios.

### 4. **CatBoost Regressor for Ball Bounce Prediction**
   - **High-Accuracy Predictions**: Predicts the bounce of the tennis ball with minimal hyperparameter tuning, critical for real-time applications like the Hawk-Eye system.

## 🖼️ Visual Components

### Mini-Map Visualization
The **Mini-Map** component provides a real-time visualization of both players' and the ball’s positions on a simplified court layout. It dynamically updates throughout the game, providing real-time insights on player positioning, ball movement, and court coverage.

### Hawk-Eye System
Whenever the ball lands outside the valid court region, the **Hawk-Eye OUT** box is displayed to highlight incorrect shots, aiding in decision-making.

### Real-Time Metrics Analysis
Key statistics like player speed, shot speed, and depth are displayed in the analysis box. Metrics include:
- **Player Speed**: Real-time player speed tracking.
- **Shot Speed**: Speed of each shot and average shot speed for both players.
- **Shot Depth**: Analysis of how deep the shots land in the opponent's court.


## 🚀 Setup Instructions

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/yourusername/RacketVision.git
   ```
2. **Navigate to the Project Directory**: 
   ```bash
   cd RacketVision
   ```
3. **Create and Activate Conda Environment**: 
   ```bash
   conda env create -n racketvision -f environment.yml
   conda activate racketvision
   ```
4. **Install Jupyter Notebook**: 
   ```bash
   conda install -c conda-forge notebook
   ```
5. **Launch Jupyter Notebook (If you haven't already)**: 
   ```bash
   jupyter notebook
   ```

## 🏗️ Future Work
- **Player Posture Estimation**: Classifying tennis shots (serve, return, volley, forehand, backhand, etc.) based on posture.
- **Shot Speed Analysis**: Speed metrics for various shots based on player posture and ball interaction.

## 🤝 Contributing
Contributions are welcome! Please feel free to fork the repository and submit pull requests for any improvements or new features.

---
