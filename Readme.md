## PAD-VO: Patch Agreement Deep Visual Odometry

### Motivation:
1. visual odoemtry calculation does not rely on the whole image. A featureful sub-region (patch) is enough for estimating an ego-motion.
2. The images patches have more similarity over scene.
3. Estimation from patches can reduce the estiamtion variance
4. Estimation from patches can increase the training data amount implicitly


### Method
1. Estimated a ego-motion from each patch together with a reliability
2. Achieve the overall ego-motion and reliability


### Paper
1. Coming soon

### To Run This Code:
#### Requirement
1. only if you can build this docker image had have at least 8G GPU
2. install all the requirement as mentioned in the dockerfile.
```
cd dockerfile
docker build -t xxx/pytorch .
```
#### Run
1. run the docker container
2. create the dataset folder for your dataset like kitti folder there;
3. modify script train.sh about for some option, for the meaning for options refer to src/option.py
4. run the code in the docker container
```
sh script/train.sh
```
### Interesting discovery
[![IMAGE ALT TEXT](http://img.youtube.com/vi/KLh6D69Ubag/0.jpg)](http://www.youtube.com/watch?v=KLh6D69Ubag "What has PAD-VO learned")
