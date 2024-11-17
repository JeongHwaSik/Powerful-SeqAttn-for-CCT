# 🔥Powerful SeqAttention for CCT🔥

For details see [Powerful SeqAttention for Compact Convolutional Transformer](https://github.com/JeongHwaSik/Powerful-SeqAttn-for-CCT/blob/main/Powerful_SeqAttn_for_CCT.pdf) by Hwasik Jeong and Jongbin Ryu.

<br>

## Abstract
As convolution and transformer architectures have advanced, performance on classification tasks has steadily improved, even surpassing human capabilities. 
During this development, the limitation of transformers being effective only with large datasets was addressed by the introduction of [CCT](https://arxiv.org/pdf/2104.05704), a hybrid version of convolution and transformer architectures. 
CCT opened up the possibility for transformers to perform well on small datasets. 
The sequence pool used in CCT achieved 76.93\% top-1 accuracy on the CIFAR-100 dataset by learning various features, but it was found to lack feature diversity to further enhance performance. 
This study reveals that [gramian attention](https://openaccess.thecvf.com/content/ICCV2023/papers/Ryu_Gramian_Attention_Heads_are_Strong_yet_Efficient_Vision_Learners_ICCV_2023_paper.pdf) is much more effective at learning diverse features compared to the original sequence pooling, achieving 80.52\% on the CIFAR-100 dataset with 8 heads. 
Taking this a step further, this study proposes a new head architecture called SeqAttention, which is much more lightweight and powerful compared to the gramian attention head.

<br>

## 🌟 Method
This study proposes a new head, called 🔥'**SeqAttention**'🔥, which outperforms both sequence pooling and gramian attention. 
This head architecture integrates sequence pooling with an attention mechanism, as illustrated below.
<p align='center'>
  <img width="674" alt="Screenshot 2024-10-16 at 9 20 17 PM" src="https://github.com/user-attachments/assets/be18a131-398d-44c8-ba4f-fee911bfe00c">
</p>

<br>

## Results
SeqAttention head in CCT (SeqAttn1-CCT-7/3x1) not only surpasses both the original CCT and CCT with a single Gramian attention head, but also does so with fewer parameters compared to GA1-CCT-7/3x1. 
The results below present the Top-1 and Top-5 accuracy on the CIFAR-100 and ImageNet datasets, along with the total number of parameters.
<p>
  <img width="1000", height="280" alt="Screenshot 2024-10-16 at 9 31 33 PM" src="https://github.com/user-attachments/assets/b3e2ff87-fac8-4789-b3cd-e8c87b06599a">
</p>

<br>

## Training & Evaluation
- CIFAR-100(optional) and ImageNet datasets should be in the following folder.
```
data/
  cifar-100-python/
    meta  
    test   # 10,000 validation images
    train  # 50,000 training images
  imageNet/
    train  # 1,281,167 training images
    val    # 50,000 validation images
```


- Train & Test on a single GPU
  
```
CUDA_VISIBLE_DEVICES=0 python3 multi_train.py cifar100 -m seqattn1_cct_7_3x1_32
```

- Train & Test on multiple GPUs

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 multi_train.py imagenet -m cct_7_3x1_32
```
