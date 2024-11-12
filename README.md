# Introduction
This repository contains the dataset and code utilized in our research on developing **HM-TDF**, a hard sample mining-based tongue diagnosis framework for fatty liver disease severity classification. 
# Paper Title
Hard Sample Mining-Based Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network
# Method
We propose a Hard sample Mining-based Tongue Diagnosis Framework (HM-TDF) for fatty liver disease severity classification, which identifies hard-to-classify samples based on uncertainty and applies targeted learning to enhance classification performance, as shown in the following figure.
![image](https://github.com/AutoTongueD/HM-TDF/blob/main/figures/framework_overview.png)
We introduce a Multi-source Feature Fusion Kolmogorov-Arnold Network (MFF-KAN) and a three-stage training strategy, as shown in the following figure, to model the relationship between tongue images plus basic physiological indicators and FLD severity.
![image](https://github.com/AutoTongueD/HM-TDF/blob/main/figures/network_and_train_strategy.png)
# Released Dataset
The dataset, named Tongue-FLD, includes 5,720 samples: 3,653 with non-FLD, 1,532 with mild FLD, and 535 with moderate/severe FLD, resulting in an imbalance ratio of 6.82/2.86/1.00. Each sample includes a segmented tongue image, FLD severity annotation and eight physiological indicators. The physiological indicators include Gender, Age, Height, Waist circumference, hip circumference (Hipline), Weight, body mass index (BMI), systolic blood pressure (SBP), and diastolic blood pressure (DBP). The data was obtained from a cohort study that received ethical approval. The participants were residents of Fuqing City, Fujian Province, China, aged 35 to 75 years. For each participant, a facial image with the tongue extended was captured using "TCM Diagnostic Devices," and basic physiological indicators were measured. Subsequently, participants underwent ultrasound examinations, and FLD severity was assessed according to the standard criteria established by the Fatty Liver Disease Study Group of the Chinese Liver Disease Association. 
![image](https://github.com/AutoTongueD/HM-TDF/blob/main/figures/data_collection_and_preprocessing.png)
# Usage
Python: 3.8.19

    pip install -r requirements.txt
    cd ./Tongue-FLD
    cat Tongue_Images.tar.gz.* > Tongue_Images.tar.gz
    tar xzf Tongue_Images.tar.gz
    python random_rotate_images.py
    python rotate_pre_train.py
    python main.py

# Cite this repository
If you find this code or dataset useful in your research, please consider citing us:  
@inproceedings{HM-TDF,  
  title={Hard Sample Mining-Based Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network},  
  link={https://github.com/AutoTongueD/HM-TDF}  
}  

# Reference
[https://github.com/KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)  
[https://github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)  
[https://github.com/AntonioTepsich/Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs)  
