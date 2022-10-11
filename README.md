# EDAE: Exponential deviation Autoencoder
By Min-Seong Kwon, Yong-Geun Moon, Byungju Lee, Jung-Hoon Noh. Autoencoders with Exponential Deviation Loss for Weakly Supervised Anomaly Detection (Pattern Recognit. Lett.)

## Brief Introduction
EDAE (exponential deviation autoencoder) is a weakly supervised anomaly detection method, which leverages a few labeled anomalies to improve the detection performance. 

EDAE exploits the estimated probability distribution of the anomaly scores, which enables to eliminate potential anomalous data from unlabeled data. 

Our proposed loss fuction called EDL (exponential deviation loss) ensures that the anomaly scores of abnormal data deviate significantly from those of normal instances. 

The resulting EDAE model achieves significantly better anomaly scoring than the competing deep methods. 

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
.

## Citation
.

## Test Environment
We used datasets and algorithms provided by [ADBench](https://arxiv.org/abs/2206.09426).

## Contact
If you have any question, please email to Prof. Jung-Hoon Noh (email: jhnoh@kumoh.ac.kr) or Mr. Min-Seong Kwon (email: 20170058@kumoh.ac.kr).
