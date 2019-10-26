# Dog Breed Classification

![Cover Picture](https://storage.googleapis.com/kaggle-competitions/kaggle/3333/media/border_collies.png)

An effective deep learning model based on VGG-16 and RESNET50 architectures to classify any dog image from 133 different classes of dogs. Added advantage, you can pass images of humans as well. It is intelligent enough to classify between a human and a dog as too.

## Usage

```
python3 script.py <FULL-PATH-TO-IMAGE-FILE>
```

**Example**

```
python3 script.py /home/mypc/Desktop/pexels-photo-356378.jpeg
```

```
IMAGE PATH:  /home/mypc/Desktop/pexels-photo-356378.jpeg
This is dog!
Your breed is most likely ... CANAAN DOG
```

### VGG-16 Architecture

![](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)


### RESNET-50 Architecture

![](https://eenews.cdnartwhere.eu/sites/default/files/styles/inner_article/public/sites/default/files/images/resnet50_630.jpg)

## Scope Of Improvement

- Classify images by passing URLs directly
- Develop a stable and robust REST API
- Current accuracy of Dog Breed Classification Sub-Model = 81%. This can be improved

