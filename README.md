# ME592Final

For questions please reach out at either ambray@iastate.edu or austinbray1999@gmail.com

Raw data files are too large for GitHub, but can be provided as needed. 

Key files within this directory include: 

Create_Data.py is technique to collect images and keystrokes from GTA.

Screengrab_Getkeys.py was the origional technique for collecting images and keystrokes in GTA. This method proved very inefficient computationally. 

Train_ModelNew allows either a CNN or a CNN+RNN to be used to train using .npz file from Create_Data.py

Run_Model_In_Game.py uses trained model to drive vehicle in GTA. 

TestingScreenYoloV5.py was utilized for image detection tasks we explored. This code is from the existing YOLO project. 

lane_assist.py us utilized to map lane lines onto images recorded playing GTA. 
