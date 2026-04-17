# Autonomous-Driving-Car-simulation-using-Pytorch🚘🚘🚗🚗

## 💥Contributors-
   * Shreyas.P.J , github: shreyaspj20

# 💥ABSTRACT :

   A self-driving car, also known as an autonomous vehicle (AV or auto), driverless car, or robo-car is a vehicle that is capable of sensing its environment and moving safely with little or no human input. In this project, we will be using camera sensors(center,left and right) implanted on the car to make predictions on the steering angle.This steering angle will be used to control the car. We will start of with building a model with the architecture which is defined in the NVIDIA reasearch paper https://arxiv.org/pdf/1604.07316v1.pdf. This model was trained on 22 epochs. The model could perform better if it is trained on a lot more training images. Augumentation of images could also help reduce overfitting of the model.

 
 # 💥IMPORTANT LIBRARIES USED :
   * OpenCV.
   * Torch.
   * Torchvision.
   * Socketio.
   * Eventlet.
   * Pillow.


# 💥SIMULATOR USED.
   To test how our model actually works, we need a simulator to actually run our trained model on. We can run simulations for such models on CARLA or Grand Theft Auto(GTA) but we will be using an open sourced simulator by Udacity. One can find the documentation and download the beta version simulator from https://github.com/udacity/self-driving-car.
   This has 2 modes:
   * Training mode: We have to generate train images by use of manual driving. The train images are generated along with the driving csv file which contains the path of each image captured and also the steering angle at each instance.
   * Autonomous mode: After we are done with training our model, we can test it by deploying the model on the simulator.

Download Simulator
- Windows：https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-windows.zip
- Mac：https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-mac.zip
- Linux：https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip
(source at https://github.com/udacity/self-driving-car-sim)


For mac:
 
```bash
APP_PATH="/Users/zhengzhedong/Desktop/课件/term2_sim_mac/term2_sim.app" # change to your path
ls "$APP_PATH/Contents/MacOS/"   # return the real name ``term2_sim_mac``
chmod +x "$APP_PATH/Contents/MacOS/term2_sim_mac"
arch -x86_64 "$APP_PATH/Contents/MacOS/term2_sim_mac"
```

![](Self_Driving_Car_Simulation.gif)

# 💥HOW TO USE :
   * Install all the required dependencies defined in the requirements.txt.
```
pip install -r requirements.txt
```
   * This application can be used by executing the run.py file.
   * The model could be run by executing this command on the terminal  "python run.py model.h5".
   * If you are using your own model, then replace model.h5 with your model's path in the command.

Enjoy the ride😊😊

