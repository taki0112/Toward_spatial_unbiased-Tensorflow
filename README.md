## Spatial unbiased GANs &mdash; Simple TensorFlow Implementation [[Paper]](https://arxiv.org/abs/2108.01285)
### : Toward Spatially Unbiased Generative Models (ICCV 2021)

<div align="center">
  <img src="./assets/teaser.png">
</div>

## Requirements
* `Tensorflow >= 2.x`

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── 000001.jpg 
       ├── 000002.png
       └── ...
```

### Train
```
> python main.py --dataset FFHQ --phase train --img_size 256 --batch_size 4 --n_total_image 6400
```

### Generate Video
```
> python generate_video.py
```

## Results
* **FID: 3.81**
<div align="center">
  <img src="./assets/sample.gif">
</div>

## Author
[Junho Kim](http://bit.ly/jhkim_resume)
