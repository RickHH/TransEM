# [TransEM:Residual Swin-Transformer based regularized PET image reconstruction](https://arxiv.org/abs/2205.04204)
## Abstract
Positron emission tomography(PET) image reconstruction is an ill-posed inverse problem and suffers from high level of noise due to limited counts received. Recently deep neural networks especially convolutional neural networks(CNN) have been successfully applied to PET image reconstruction. However, the local characteristics of the convolution operator potentially limit the image quality obtained by current CNN-based PET image reconstruction methods. In this paper, we propose a residual swin-transformer based regularizer(RSTR) to incorporate regularization into the iterative reconstruction framework. Specifically, a convolution layer is firstly adopted to extract shallow features, then the deep feature extraction is accomplished by the swin-transformer layer. At last, both deep and shallow features are fused with a residual operation and another convolution layer. Validations on the realistic 3D brain simulated low-count data show that our proposed method outperforms the state-of-the-art methods in both qualitative and quantitative measures.
## Flowchart
![pic](https://github.com/RickHH/TransEM/blob/main/Method.png)
## Contact
Feel free to contact rickhu@zju.edu.cn if you have any questions.
## Acknowledgments
[FBSEM](https://github.com/Abolfazl-Mehranian/FBSEM)
