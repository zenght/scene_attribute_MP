# Learning scene attribute for scene recognition
The paper has been accepted by IEEE transcations on Multimedia. [[paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8851274).

This is code of **Learning scene attribute for scene recognition**, our code is mainly implemented by Tensorflow, the pytorch code is also available (just need to change the shape of features). If you want to use our method, we have packaged our code, you only need to modify the corresponding feature path in the configs. We optimize the features by hierarchical message passing, and use these optimized features to train LinearSVM to conduct scene recognition.

# Evaluation results on MIT67 and SUN397
|Model|MIT67(%)|SUN397(%)|
|:---|:---|:---|
|ImageNet(IM)|81.04|64.15|
|Attribute-ImageNet(AI)|82.31|63.82|
|Places(PL)|85.3|69.7|
|Attribute-Places(AP)|85|69.66|

|Method|MIT67(%)|SUN397(%)|
|:---|:---|:---|
|IM+PL|87.01|72.03|
|AI+PL|87.24|72.21|
|PL+AP+AI+IM|87.46|72.41|
|IM+PL(w.s)|87.39|73.81|
|AI+PL(w.s)|87.84|73.85|
|PL+AP+AI+IM(w.s)|88.06|74.12|
