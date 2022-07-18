# Feature Matching - Geolocalisation
This section contains the final part of this work’s system, the geolocalisation. After
extracting the roads found in the image taken by the drone, the image needs to be georeferenced with a road map. The
area in which the drone is operating is the Isle of Mull. Then, the results of applying the
feature-based algorithm for matching the two images will be fully described together
with the obstacles that occurred. Finally, the coordinates of the flying drone will be
obtained together with the error which will be implemented by using the Pythagoras
calculation.

## Map Extraction
The first part of this task is the process of extracting the appropriate map. The starting
point where the drone is going to take-off is known. Thus, a map of a hemisphere
with the drone as its centre will be given. However, which is the optimal radius for
this hemisphere? This question is fundamental as we do not want the drone to fly
outside of the selected map, and we also do not want our map to be extensive, which
will lead to worst image matching accuracy. Thus, for answering this question, firstly,
some information regarding the used drone is going to be given. The drone that has
been selected has a maximum speed of 20.117m/s and can fly up to an altitude of 500
meters. Furthermore, all the images have been taken in a vertical viewpoint.

All the maps have been taken from the OpenStreetMap (OSM) (https://wiki.openstreetmap.org/wiki/Main_Page)
and by using the Quantum Geographic Information System (QGIS)version 3.14.1 (https://qgis.org/en/site/), all the appropriate
roads have been extracted. QGIS is a well known open-source geographic information
system (GIS) application that enables the viewing, editing, and analysis of geospatial
data. While using QGIS on the extracted map from OSM the keys "highway" and
’building’ have been used in order to get all the available roads and buildings from the
map. Afterwards, a square image containing the hemisphere with the calculated radius
is extracted. Finally, by using MATLAB, the aforementioned map has been converted
to binary with black as the background and white as the roads and buildings.


The map calculated for the first image taken by the drone. The initial
images have been covert to binary images :
![Drone1](https://github.com/TypEktor/Aerial-Image-Geolocalisation-Using-Road-Detection/blob/main/Code/Feature%20Matching/Images/Drone1.jpg?raw=true)

## Image Comparison
By having a roadmap image extracted by the road detection network of part one, and a
map containing a wider area from the map extraction process, it is time to move to an
essential procedure, the image matching stage. This task, by taking two binary images
as input, tries to find as many as possible feature matches between them and draw a
polygonal curve around the area that the algorithm believes that the image has been
taken.
For this purpose, the ORB algorithm together with RANSAC, has been used where
ORB extract all the feature points and RANSAC reduces the outliers. The implementation
of both methods has been done using Python3 programming language and using
OpenCV library

One of the cases that have been tested here was using Roadmaps together with Buildings. For that purpose, the YOLOv3 detector is going to be
used. YOLOv3 has been trained using a dataset from EDINA with 325 images containing
buildings. Therefore, the first aerial image, taken by the drone, is going to be
scanned for building using this detector. After the succeeding detection of the buildings,
the image has been converted to binary and merged with the image containing the roadmap.

The image matching result of ORB and RANSAC algorithm for perfectly
labelled merged image. The matching between the points are illustrated as blue lines,
and the area in which the image has been taken as a blue polygonal curve:
![Drone2](https://github.com/TypEktor/Aerial-Image-Geolocalisation-Using-Road-Detection/blob/main/Code/Feature%20Matching/Images/Drone2.png?raw=true)

## Finding the Drone’s Coordinates
The final task of this dissertation’s system is the identification of the camera’s location
and orientation. All the images from the EDINA dataset have been taken under
a vertical view. Thus, the orientation of the camera in this work is considered to be
only vertical, and thus we do not deal with oblique views. For extracting the coordinated,
for locating the drone, a large number of equations and formulas has been used. These coordinates have been extracted by using the polygonal curve drawn in
the image matching process.

Firstly, by using the aforementioned curve, the centroid of the polygon has been
calculated using a Python script. Furthermore, by having the polygon’s centre, the next
step is to convert this point into a WGS 84 reference coordinates. As the coordinates of
the extracted map were known, this process was just a rule of three calculation, and thus
the latitude and longitude were extracted. Using the above process to the final example,
of the roadmap and buildings case, from the previous subsection, the calculated values
for latitude and longitude were 56.576089 and-6.181153, respectively. The difference
with the initial (56.577199,-6.184718) was 0.00012 and 0.00070.
Finally, as this difference does not provide much insight into the error, a conversion
to meters is essential. As we are dealing with short distances, the Pythagoras calculation
can provide a good solution. Latitude and longitude have different formulas for
their calculation. It is given that one degree in latitude is just 111.32km,

$arc_{Lon}=\frac{2\pi(R + h)}{360^{\circ}}$

where R is the world’s radius, equal to 6378km, and h is the altitude, in our case zero.
If $Lat = \phi$ and $lon = \lambda$, then:

$\Delta x = arc\cdot cos(\phi)\cdot \Delta \lambda$

$\Delta y = arc\cdot \Delta \phi$


where Delta Values defined as the difference between two points. Thus the error is
the Pythagoras of both Delta values, $err = \sqrt{\Delta x^2 + \Delta y^2}$ , which in our occasion is 79 meters.

## Implementation
