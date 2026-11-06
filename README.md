# Neural Networks Temporal Reachability Prediction
### 1. Implemented models:
> TGAT
> 
> TGN
>
> CAW-N
>
> GraphMixer
>
> DyGFormer
>
> TPNet
> 
> ReaCH-TGN


### 2. Evaluation
> - Dataset: CollegeMsg, soc-sign-bitcoinalpha, soc-sign-bitcoinotc
> - Seed: 1, 2, 3
> - Learning Rate: 0.0001, 0.0005
> - Batch Size: 200

#### 2-1. CollegeMsg
| Model | ACC | AP | AUC | 
|:---:|:---:|:---:|:---:|
| TGAT | 0.8009 | 0.8722 | 0.8721 |
| TGN | 0.9256 | 0.9605 | 0.9673 |
| CAW-N | 0.8686 | 0.9324 | 0.9183 |
| GraphMixer | 0.7891 | 0.8886 | 0.8541 |
| DyGFormer | 0.7943 | 0.8900 | 0.8701 |
| TPNet | 0.8497 | 0.9325 | 0.9229 |
| ReaCH-TGN | 0.8582 | 0.9418 | 0.9487 |

#### 2-2. soc-sign-bitcoinalpha
| Model | ACC | AP | AUC | 
|:---:|:---:|:---:|:---:|
| TGAT | 0.6598 | 0.7752 | 0.7845 |
| TGN | 0.7241 | 0.8071 | 0.8279 |
| CAW-N | 0.6154 | 0.7763 | 0.7607 |
| GraphMixer | 0.7743 | 0.8574 | 0.8523 |
| DyGFormer | 0.6230 | 0.7644 | 0.7658 |
| TPNet | 0.7564 | 0.8426 | 0.8427 |
| ReaCH-TGN | 0.6606 | 0.7739 | 0.8005 |

#### 2-3. soc-sign-bitcoinotc
| Model | ACC | AP | AUC | 
|:---:|:---:|:---:|:---:|
| TGAT | 0.6413 | 0.7658 | 0.7688 |
| TGN | 0.7932 | 0.8552 | 0.8799 |
| CAW-N | 0.6667 | 0.8203 | 0.8030 |
| GraphMixer | 0.7728 | 0.8713 | 0.8668 |
| DyGFormer | 0.6707 | 0.7816 | 0.7846 |
| TPNet | 0.7647 | 0.8641 | 0.8633 |
| ReaCH-TGN | 0.7025 | 0.8140 | 0.8328 |
