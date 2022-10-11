# EDAE: Exponential deviation Autoencoder
By Min-Seong Kwon, Yong-Geun Moon, Byungju Lee, Jung-Hoon Noh. Autoencoders with Exponential Deviation Loss for Weakly Supervised Anomaly Detection

## Brief Introduction // 수정필요
EDAE(Exponential deviation autoencoder) will be introduced in Pattern Recognition Journal, which leverages a limited number of labeled anomaly data and a large set of unlabeled data to perform end-to-end anomaly score learning. 

It addresses a weakly supervised anomaly detection problem in that the anomalies are partially observed only and we have no labeled normal data.

Unlike other deep anomaly detection methods that focus on using data reconstruction as the driving force to learn new representations, DevNet is devised to learn the anomaly scores directly. 

Therefore, DevNet directly optimize the anomaly scores, whereas most of current deep anomaly detection methods optimize the feature representations. 

The resulting DevNet model achieves significantly better anomaly scoring than the competing deep methods. 

Also, due to the end-to-end anomaly scoring, DevNet can also exploit the labeled anomaly data much more effectively. 

## Usage
A simple example of running our proposed method is shown in `EDAE_test.ipynb.`

See `./baseline/EDAE/run.py` for more details about each argument used in this line of code.

The key packages and their versions used in our algorithm implementation are listed as follows
* python==3.9.12
* keras==2.8.0
* tensorflow-gpu==2.8.0
* scikit-learn==1.1.1
* numpy==1.21.5
* pandas==1.4.3
* scipy==1.7.3

See the full paper for the implemenation details of EDAE.

## Full Paper
The full paper can be found in IEEE Xplore or [arXiv](https://arxiv.org/abs/2105.10500). // 수정 필요

## Citation
> Yingjie Zhou, Xucheng Song, Yanru Zhang, Fanxing Liu, Ce Zhu and Lingqiao Liu. Feature Encoding with AutoEncoders for Weakly-supervised Anomaly Detection, IEEE Transactions on Neural Networks and Learning Systems, 2021. // 수정 필요

## Test Environment
We used datasets and algorithms provided by [ADBench](https://arxiv.org/abs/2206.09426).

## Contact
If you have any question, please email to Prof. Jung-Hoon Noh (email: jhnoh@kumoh.ac.kr) or Mr. Min-Seong Kwon (email: 20170058@kumoh.ac.kr).
