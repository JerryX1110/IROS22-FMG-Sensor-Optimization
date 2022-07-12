# [Official] '[Optimization of Forcemyography Sensor Placement for Arm Movement Recognition]()'

![image](https://user-images.githubusercontent.com/65257938/178545143-3e3919b1-9b67-4ed8-8288-dc1c6e4a39e9.png)


>**Optimization of Forcemyography Sensor Placement for Arm Movement Recognition**<br>
>Xiaohao Xu, Zihao Du, Huaxin Zhang, Ruichao Zhang,
Zihan Hong, Qin Huang, and Bin Han∗, Member, IEEE

>**In the Proceedings of the 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS22)**

>**Abstract:**  *How to design an optimal wearable device for human movement recognition is vital to reliable and accurate human-machine collaboration. Previous works mainly fabricate wearable devices heuristically. Instead, this paper raises an academic question:can we design an optimization algorithm to optimize the fabrication of wearable devices such as figuring out the best sensor arrangement automatically? Specifically, this work focuses on optimizing the placement of Forcemyography (FMG) sensors for FMG armbands in application of arm movement recognition. Firstly, the armband is modeled based on graph theory with both sensor signals and connectivity of sensors considered. Then, a Graph-based Armband Modeling Network(GAM-Net) is introduced for arm movement recognition. Afterward, the sensor placement optimization for FMG armbands is formulated and an optimization algorithm with greedy local search is proposed. To study the effectiveness of our optimization algorithm, a dataset for mechanical maintenance tasks using FMG armbands with 16 sensors is collected. Our experiments show that a comparable recognition accuracy with all sensors can be 
maintained even with 4 sensors optimized with our algorithm. Finally, the optimized sensor placement result is verified from a physiological view. This work would like to shed light on the automatic fabrication of wearable devices with downstream tasks, like human biological signal collection and movement recognition, considered.*

## Prerequisites
### Recommended Environment
* Python 3.7
* Pytorch 1.7
* CUDA 10.1

### Depencencies
You can set up the environments by using `$ pip3 install -r requirements.txt`.

### Data Preparation
1. Download our self-collected FMG dataset on mechanical maintenance tasks.
- BaiduDisk Link: https://pan.baidu.com/s/1NYCBs1VkBx20i-INAJZZ2w 
- BaiduDisk Password:sga8.

2. Place the `original_data_new` inside the working folder.
- ensure the data structure is as follows.
~~~~
|-original_data_new
    |-dzh
        |-data_2022-01-07-21-22-13.csv
        ...
        |-data_2022-01-07-21-44-10.csv
    |-hzh
    |-xxh
    |-zhx
    |-zrc
~~~~

3. Preprocess data.
- You can preprocess data by running the script below.
```python
python data_preprocess.py
```
- After preprocessing,`window_data_new` will be created.

## Usage
### Training of Recognition Model
- For comparison with our **GAM-Net**, three widely-used strong action recognition models, including Long short-term memory (LSTM), Artificial Neural Network (ANN), and Convolutional Neural Network, were employed. You can easily train and evaluate the model by running the script below.
```python
python main.py
```
- Before train the model, you can change model_name and subject_index to decide which model to train and which subject's data to use. 

### Sensor Placement Optimization Algorithm
You can run the script below to run the Sensor Placement Optimization algorithm.
```python
python SPO.py
```

### Result Visualization and Analysis
You can run the script below to visualize all of the above results.
```python
python result_visualize.py
```


## Citation
If you find this work is useful for your research, please consider citing:

 ```latex
@inproceedings{xu2022optimization,
  title={Optimization of Forcemyography Sensor Placement for Arm Movement Recognition},
  author={Xu, Xiaohao and Du, Zihao and Zhang, Huaxin and Zhang, Ruichao and Hong, Zihan and Huang, Qin and Han, bin}, 
  booktitle={Proceedings of the 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS22)},
  year={2022}
}
```

if you find the implementations helpful, please consider to cite:

 ```latex
@misc{xu2022optim,
  title={FMG-Optimiz-REPO},
  author={Xiaohao Xu and Huaxin Zhang},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/JerryX1110/IROS22-FMG-Sensor-Optimization/}},
  year={2022}
}
```

## Contact
If you have any question, feel free to report a issue in this repo or directly contact the author _Xiaohao XU_(xxh11102019 [AT] outlook.com).
