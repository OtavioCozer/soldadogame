# Soldado Game

This is a AR game written in python using PyGame, OpenGL and OpenCV.

https://github.com/OtavioCozer/soldadogame/assets/38982699/033d94bf-fbd7-481f-9fb0-4e7e0bc78b46

## Usage

You'll need to have **python**, **pip** and **virtualenv** installed.

First, you need to start python virtual enviroment and download the required dependencies. For this, use `prepare_env.sh`:

```sh
source prepare_env.sh
```

Now, you can run `pygame_soldado_ar.py` either with your own webcam or using then video file `video.mp4` provided.

### Video File:

Using with video file is straightforward, just run:

```sh
python3 pygame_soldado_ar.py -v video.mp4
```

### Webcam:

If you want to run with your own webcam, you'll need to calibrate your camera. On the calib folder there is several pictures that I used to calibrate my own webcam. Take similar pictures with yours and then run the calibration script:

```sh
python3 camera_calib.py
```

Use your printer to print the files inside **_aruco folder_**. Now you can run the game. Make sure both aruco patterns that you printed are in the webcam field of view, move then with your hand to avoid getting hit by the other player, press **KEY L** or **KEY S** to shoot.

```sh
python3 pygame_soldado_ar.py
```
