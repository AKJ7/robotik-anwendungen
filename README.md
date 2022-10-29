# Robotik Anwendungen  

Codes for my Robotik Anwendungen project of 2021: Bartender robot.

## Tasks  
- Robot should be able to serve drinks (a.k.a beers) to customers
- Robot should be able to continuously measure the fill level of glass content  


## Synopsis  
- The project took place during the Winter semester 2021/2022.  
- 3 Teams, each of 7 students competed to provide the best results.
- Being part of the first team, my task consisted of the continuous detection of the fill level.
- To measure the filling level of the glass, both the level of the beer foam/crown and the 
beer itself needed to be measured
- Team 2 opted for a system based on a capacitive and ultrasonic sensors.
- Team 3's idea was based on mixture weight sensor and an ultrasonic sensor. 
- With the ultrasonic sensor, the uppermost part of the content of the glass was to be measured.
- With the capacitive sensor, placed on the palm of the robot's grip, only the contact area of 
the sensor to the glass could be measured when activated by the filled drink.
- The weight sensor helped by returning the weight of the glass's content, from which its volume
was approximated.
- Both implementations were comprises:
  - Foam detection only worked on very tick and full foam else ultrasonic waves would propagate 
  - Ultrasonic sensor needs to be place perfectly perpendicularly to the glass
  - Team 2 couldn't measure the fill level continuously, rather returned a boolean value
  - Team 3's method to calculate the fill level was glass and drink dependent
  - The ultrasonic sensor could not be used for any drink without foam
- Our implementation was based on the usage of a camera and technics from computer vision to detect 
the level of the beer and its foam. It had multiple advantages:
  - Only one sensor was needed: the camera
  - The beer crown/foam could be measured independent of state and shape - i.e. worked when no crown was present
  - No hardware dependencies
  - Could be used to detect multiple types of drinks at once
- Our technic was not perfect:
  - A powerful processing unit was needed
  - Knowledge of computer vision was required.  


## Installation  

The Camera used is that of my phone connected using the `IP Camera` [Android app](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=de&gl=US).  

----

The code contains codes to detect the filling levels and show the camera stream - for debugging
purposes.  
The installations require pipenv. Install one just in case using:  
```bash 
pip install pipenv
```
- Install dependencies using:
```bash
pipenv update
```
- Run the detector:  
```bash
pipenv run capture
```
- Run the camera stream viewer:
```bash
pipenv run camera
```  

----
Build the docs. Requirement: `docker`.
```bash
cd docs && docker run --rm -it --user="$(id -u):$(id -g)" -v "$(pwd)":/home aggipp/texlive latexmk
```  

## TODO  
Corners were cut to save time. The following can still be implemented as improvements:  
- Only search for the levels inside the glass 
- Increase performance