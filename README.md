# scene attribute Message passing
This is code of <Learning scene attribute for scene recognition>, our code is mainly implemented by Tensorflow, the pytorch code is also available (just need to change the shape of features). Since the paper is still under review, the method might be not easy to undedstand. But if you want to use our method, we have packaged our code, you only need to modify the corresponding feature path in the config.   

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
|IM+PL(w.s)|87.39|73.81|
|AI+PL(w.s)|87.84|73.85|
