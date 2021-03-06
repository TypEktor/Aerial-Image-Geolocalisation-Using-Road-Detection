# Aerial-Image-Geolocalisation-Using-Road-Detection

This repository contains the work that has been done as part of my Masters's Thesis at the University of Manchester. There has been an effort to locate a moving drone using high-resolution aerial images by observing roads and buildings. For that purpose, a plethora of different Fully Convolutional and Convolutional Neural Networks (CNN) has been used for the image segmentation work (road detection) and the algorithm YOLO has been used for the object detection part (building detection). Finally, a combination of feature-based image matching algorithms has been used to locate the drone, against a given map.


The following picture depicts the abstract of that Thesis

![Thesis Abstract](https://github.com/TypEktor/Aerial-Image-Geolocalisation-Using-Road-Detection/blob/main/Abstract.jpg?raw=true)

---

Diving into the three parts of the aforementioned Dissertation:

## Object Detection

Besides roads, buildings can also be considered an effective way to georeference an aerial image, taken from a UAV, with a referenced map. This work focuses on road
detection; however, roads and buildings may cooperate and produce better results. The YOLO family of networks have been proven to be very accurate while dealing with
features in aerial images, and thus, these methods have been chosen for this work’s system.

## Road Detection

Deep neural networks can be considered a reliable approach in order to deal with semantic image segmentation problems. As already cited, the first task of this work is
to effectively extract roads from high-resolutional aerial images. This subsection focuses on describing all the implemented deep neural networks used for achieving the aforementioned task. Specifically, three models have been used, an original U-Net architecture, a Residual U-Net model, and a Multi-Residual U-Net.


## Feature Matching

The final task of this dissertation is focused on georeferencing a roadmap extracted by the previous deep neural networks with a roadmap derived from a map. In order to achieve this objective, a feature-based image matching approach has been used. This approach combines two algorithms, firstly the Oriented FAST and Rotated BRIEF is
used for the extraction of feature points and feature quantities and then for increasing the reliability of matching, the Random Sample Consensus is applied.



---

## Contents
- **Object Detection:** A folder containing the code and the output for the Object Detection part of the Thesis
- **Road Detection:** A folder containing the code and the output for the Road Detection part of the Thesis
- **Feature Matching:** A folder containing the code and the output for the Feature Matching part of the Thesis


---

## Data
This work aims to achieve a matching between a drone’s camera view and a map, from a specific remote area. The training data for road detection were extracted from EDINA’s Digimat Service (https://digimap.edina.ac.uk). EDINA is a centre for digital expertise, based at the University of Edinburgh as a division of Information Services. The dataset covers images from the area of Scottish Islands such as the Isle of Mull, the Eigg island, and Tiree island, taken from 2016 and 2019. The data from EDINA contain mainly rural, mountainous, remote woodland areas and the roads are not easily distinguished. Furthermore, occasionally there are buildings, ports and warehouses. Different terrains have their characteristics which may make the road extraction process more laborious and more complicated. The extracted data are RGB, high-resolutional vertical aerial images with 25cm pixel resolution, and of scale 1 : 500. The dataset consists of 56 images, all of which are 40004000 pixels, see Figure 4.1. The goal is to extract roads from these images.


As EDINA’s data are not already labelled, each image’s mask was hand-labelled using MATLAB’s ver. R2020a image segmenter application2. The mask is given
in a grayscale format, with white standing for road pixel and black standing for the background. Due to that labelling process, the ground truth images are not perfect. Specifically, the image segmenter application is a semi-automatic labelling extension which is used for easy distinctive objects. It this dataset’s occasion, in which most of the roads are covered with dirt and soil, the application didn’t have quality results. Thus, especially the road edges are not well labelled. A reasonable question would be, why not extract the labels from OpenStreetMap (OSM)3. The reason is that in such remote areas, like in our dataset, even the OSM lack most of the roads. There are numerous sideways, and back-roads that are not labelled and only highways and roads in urban areas are tagged. Thus, for measuring the performance of the

For copyright reasons I won't be able to upload the dataset on GitHub. However, I will use two or three zoom-in pictures from the intitial dataset as examples.

---

## Pre-requisites
For the Object Detection section, a .yml file is provided for loading the environment into Anaconda. However, as all the other sections have been created into Google Colab, there is not a requirement.txt or a yml file that can be given.

---

## Resources

---

## Copyrights
- The author of this thesis (including any appendices and/or schedules to this thesis)
owns certain copyright or related rights in it (the “Copyright”) and s/he has
given The University of Manchester certain rights to use such Copyright, including
for administrative purposes.

- Copies of this thesis, either in full or in extracts and whether in hard or electronic
copy, may be made only in accordance with the Copyright, Designs and Patents
Act 1988 (as amended) and regulations issued under it or, where appropriate,
in accordance with licensing agreements which the University has from time to
time. This page must form part of any such copies made.

- The ownership of certain Copyright, patents, designs, trade marks and other intellectual
property (the “Intellectual Property”) and any reproductions of copyright
works in the thesis, for example graphs and tables (“Reproductions”), which
may be described in this thesis, may not be owned by the author and may be
owned by third parties. Such Intellectual Property and Reproductions cannot
and must not be made available for use without the prior written permission of
the owner(s) of the relevant Intellectual Property and/or Reproductions.

- Further information on the conditions under which disclosure, publication and
commercialisation of this thesis, the Copyright and any Intellectual Property
and/or Reproductions described in it may take place is available in the University
IP Policy (see http://documents.manchester.ac.uk/DocuInfo.aspx?
DocID=24420), in any relevant Thesis restriction declarations deposited in the
University Library, The University Library’s regulations (see http://www.library.
manchester.ac.uk/about/regulations/) and in The University’s policy on
presentation of Theses
