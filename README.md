The origininal dataset RSNA Pneumonia requires locating the lung opacity region in chest X-ray.\
We adapt this dataset for a binary classification of Normal vs Lung Opacity.\
We train a classifier, then test its performance on locating the anomaly region.

We studied 2 protocols to assess systematically the attention heatmaps generated by GradCAM.\
The 1st protocol assess the trustworthiness of the attention heatmap with proportion inside bounding box (follwing GradCAM).\

<img src="/img/heatmap-jzech-box.png" width="300">
Attention weight matrix and bounding box <br>
Modified from https://jrzech.medium.com/what-are-radiological-deep-learning-models-actually-learning-f97a546c5b98


The 2nd protocol measure trustworthiness by Intersection Over Union of annotated and predicted bounding box.\
The predicted bounding box is defined as the smallest rectangle holding at leats 95% of activation (following ProtoPNet).\
![](/img/iou-best1-0652.png)
Left: target bounding box in red. Middle: heatmap. Right: rectangle holding at least 95% of activation in orange. IOU = 0.65

The Intersection Over Union protocol appears to be the better one among the two.\
Nevertheless, this protocol still has some flaws. \
The major problem to address is how to convert attention heatmap to predicted bounding box.\
![](/img/iou-noncontiguous-0313.png)
Failure with multiple non-contiguous regions

Instructor: Isabelle Guyon, Kim Gerdes
Université Paris Saclay 2021


References:
GradCAM: https://arxiv.org/abs/1610.02391 \
ProtoPNet: https://arxiv.org/abs/1806.10574 \
RSNA Pneumonia Data: https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
