
# MachineVision
All work for the course Machine Vision on the Faculty of Computer Science &amp; Engineering.

Working with main library OpenCV2

LAB 01 - 07 updated and finished in the repository

Project Documentation: 

## Go kitesurfing or waste a day on the beach
### Kitesurfing sessions labeling with help of machine vision

#### Motivation

Kitesurfing, kiteboarding, kite foiling, windsurfing and a lot of variations of these sports are highly dependable on the weather especially on a single component, the WIND.
Having the knowledge of the best windy spot for the day is highly pursued by all the fanatics and newcomers the sports. Traditionally only with following the usual forecast for the top 100 known locations there can be a likely, key word is likely, accurate representation of the conditions for the next day and if there is the usual circulation of pressures in the atmosphere, then a 5-6 days forecast could be pretty accurate. Having the exact wind speed predicted on an hourly basis is still challenging and that’s why these sports are usually planned for vacation days when the rider could afford to spend a couple of hours on the beach waiting.
An approach that I believe is a solution to this problem is localization with IoT devices. Having all the data from the different spots measured locally will bring us closer to having accurate predictions even earlier than now and even there is the possibility of even closer predictions on an hourly basis.
But, how do we know if the conditions are met? How to choose an appropriate gear for the occasion? Are all riders professional forecast readers? Do they know how their gear will behave on all conditions?
That is why I am introducing Machine Vision to the solution. With machine vision tracking of the wings and the riders can be accomplished, which can be used for labeling of the gear used during the sessions, based on the conditions and also even more valuable is the information of getting an information if even there was a session taking place. These info is especially needed in places that have way less visitors and experienced locals that could tell you when riding is possible.




#### Introduction

This solution focuses on recognizing a successful session in real time. A machine vision algorithm is used to detect a rider and his gear (kite, foil, wing etc.). In the current version segmentation is used to get these objects and the points from those objects are matched with the good points for tracking. Those points are tracked in the video stream and a relative speed and distance is measured on each of the object recognized. Based on a successful start of riding a timer for a session is turned on and if there is continuous riding during a period of at least 15min that session is labeled as a SUCCESS.

#### Instructions

  - Requirements
    -	Static camera with a video stream available
    -	Microcontroller
    -	Python or C
    -	OpenCV library installed with opticalflow available

  - Installation

    The camera needs to be positioned on X meters of distance from the beach (The distance will be determined in version 2. The software of v1 was tested on different videos         downloaded from YouTube).
    The code needs to be installed on the microcontroller. Current v1 is written in Python and can be easily installed on a Raspberry Pi which is likely to be the No.1 gear         since it can have a 1080p camera and different sensors that are needed for the ecosystem. The cost with this setup is on the low end.
    The data can be saved on an internal storage (with RaspPi. It can be a usb drive or memory card) or it can be sent by any kind of connection to a server.

#### Procedure


Frame from raw footage:

##### Segmentation of relevant objects

Two approaches were tested, getting the threshold with fixed values and with adaptive threshold. The second approach was very inefficient and the results from the first approach were promising and yielded good results.

```python
slika = cv.resize(slika, (1280, 720), interpolation=cv.INTER_CUBIC)
slika_gray = cv.cvtColor(slika,cv.COLOR_RGB2GRAY

_, thresh = cv.threshold(slika_gray, 40, 250, cv.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
thresh = cv.dilate(thresh, kernel, iterations=2)
```


Each frame from the stream is resized to 720x1280. The frame is converted into a grayscale img.
That grayscale image is sent to <i>cv::threshold</i> and on later dilatation is done to the threshold matrix.
 
##### Getting contours and relevant areas

```
contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

kites = []
humans = []
for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    rect_area = w*h
    rect_are = cv.contourArea(cnt)
    diff = height - y
    if y > height/2:
        rect_are = rect_are - (rect_are/ (np.power(1.5, diff)))
    else:
        diff = height/2 - y
        rect_are = rect_are + (diff/2)
```

After getting the contours from  <i>cv::findContours</i> a rectangle is formed on the boundaries of each contour and also we are taking the area that the contours are occupying. The areas would serve us to distinguish the riders and the kites. The areas are proportionally compressed or enlarged by the distance.
Discriminating the rider and the wing

```python
if 100 < rect_are < 2000:
    # print(rect_area)
    if draw_objects:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # rect = cv.minAreaRect(cnt)
        # rectPoints = cv.boxPoints(rect)
        # rectPoints = np.int0(rectPoints)
        # cv.drawContours(slika, [rectPoints], -1, (0, 255, 0))
    # RELEVANT FOR SIZE COMPARISON. SMALLER IS HUMAN LARGER IS KITE
    if rect_area < 200:
        if only_riders:
            if len(humans) > 0:
                y_rel = humans[-1]['y']
                if y + 20 < y_rel:
                    humans.append({"area": rect_area, "x": x, "y": y, "width": w, "height": h})
        else:
            # Call ML model to classify human obj in that part of the frame
            humans.append({"area": rect_area, "x": x, "y": y, "width": w, "height": h})
    else:
        # Call ML model to classify kite obj in that part of the frame
        kites.append({"area": rect_area, "x": x, "y": y, "width": w, "height": h})
```

	
Having the kites and riders separated by other irrelevant objects enables the possibility of separating the riders and the kites. In version 1 the idea was to separate them with the area of their bounding rectangles, but inconsistency of the segmentation of the objects drew me away from this idea. This might be improved with better, more advanced segmentation. 
An idea that I trust and I believe it will be successful is developing a ML model such as CNN and feed it with the area of the picture where the segmented object resides in order to classify whether that object is a wing or a rider. This is the idea for v2.
Knowing the wing size is crucial for optimization of the model. The idea was after getting the contours and calculating the area of the wing, which is always way larger than the rider, we can determine a range of sizes of the gear for that current session.
For version 1 this separation is irrelevant since we only need to track the movement of any of the riders and the wings to determine the traveling speed and the distance. 

##### Optical flow and why Lucas Kanade

Every 5 frames there is a search for new possible relevant points for tracking.

```python
feature_params = dict(maxCorners = 200,
                    qualityLevel = 0.4,
                    minDistance = 10,
                    blockSize = 2)

# Detect the good features to track
p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
if p is not None:
    # If good features can be tracked - add that to the trajectories
    p_reshaped = np.float32(p).reshape(-1, 2)
    for x, y in p_reshaped:
        for rider_obj in riders:
            x_rider = rider_obj['x']
            y_rider = rider_obj['y']
            height = rider_obj['height']
            width = rider_obj['width']
            if x_rider < x < x_rider + width and y_rider < y < y_rider + height:
                point_added = False
                trajectories.append([(x, y)])
                break
```

<i>cv::goodFeaturesToTrack</i> is used to get points from the segmentation of the frame that satisfy the parameters defined. ( Having the mask segmentation for our objects is way better than using all the pixels from the photo )
Those points that are located on the rider object, idea for v2, are appended to the trajectories that we are following. (currently in v1 the points can be located on the rider or on the wing).

```python
lk_params = dict(winSize  = (15, 15),
                maxLevel = 1,
                criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.03))

if len(trajectories) > 0:
    img0, img1 = prev_gray, frame_gray
    p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
    p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1
```

The search of moving points is done by <i>cv::calcOpticalFlowPyrLK</i> for every frame. 

##### Determining rider travel time and speed

In order to have an object starting point there was the need to group points from the same object. The groups are formed where the first point of the object is being tracked and a starting time and point is recorded. While the object is moving in the same direction the latest point is being recorded for the object’s point group. 
 
```python
# Get all the trajectories
for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
    if not good_flag:
        continue
    trajectory.append((x, y))
    # DO THIS EVERY 20 frames
    if len(trajectory_groups) > 0:
        create_group = True
        for trajectory_group in trajectory_groups:
            last_x, last_y = trajectory_group[-1]
            if last_x - 15 < x < last_x + 15 and last_y - 15 < y < last_y + 15:
                first_x, first_y = trajectory_group[1]
                dist = np.sqrt(np.power((first_x-last_x), 2) + np.power((first_y-last_y), 2))
                dist_new = np.sqrt(np.power((first_x-x), 2) + np.power((first_y-y), 2))
                if dist_new < dist:
                    if dist > w / 3:
                        t = time.time() - trajectory_group[0]
                        # print("Traveling time :", t)
                        travel_speed = dist / t
                        # print(travel_speed)
                        if travel_speed > 5:
                            if not SESSION_SARTED:
                                SESSION_SARTED = True
                                print("START OF SESSION")
                                cv.putText(img, 'START OF SESSION', (500, 50), cv.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)
                    if dist_new - dist > 2:
                        trajectory_group = [time.time(), (x, y)]
                        create_group = False
                        break
                trajectory_group.append((x, y))
                create_group = False
                break
        if create_group:
            trajectory_groups.append([time.time(), (x, y)])
    else:
        trajectory_groups.append([time.time(), (x, y)])
    if len(trajectory) > trajectory_len:
        del trajectory[0]
    new_trajectories.append(trajectory)
    # Newest detected point
    cv.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
```

After there has been a change in direction the time and starting point for the group is being reset.           
If the distance measured was more than 1/3 of the width of the frame a calculation is being made for the traveling speed of the object. (When camera information is gathered by vendor on recording radius and the cameras are statically installed on a specific distance from the water line we can determine exact distance and speed) In version 1 1/3 of the width is considered as enough distance of object travel and with knowing the results for speed since the testing videos were all from successful sessions the speed variable was determined for this condition.
 
When this condition is met and there is at least one object that has traveled the distance with this relative speed, the session is considered as started
 
After a start of a session, time of the session is being recorded and the algorithm is started on an interval of 5min to determine if the riders are still enjoying the windy conditions. If the session lasts for 15min then this session can be labeled as <b>SUCCESSFUL</b>.

#### Extraction of useful parameters

After successfully tracking the riders and kites and a working algorithm to determine a successful session there can be much improvements on the usage of machine vision for this problem. For v2 lots of features can be extracted that will provide useful information adding up on the IoT measurements pre, during and after the session. Wing sizes can be extracted, travel distances can be extracted both for a session and for a separate rider gaining important info for the gear needed on those conditions.

