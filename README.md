## Spatial bias in GANs &mdash; Simple TensorFlow Implementation [[Paper]](https://arxiv.org/abs/2108.01285)
### : Toward Spatially Unbiased Generative Models (ICCV 2021)

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

## Author
[Junho Kim](http://bit.ly/jhkim_resume)
