# AKSVD: Attentive Deep K-SVD Network

**ATTENTIVE DEEP K-SVD NETWORK FOR PATCH CORRELATED IMAGE DENOISING**

![process_1](assets/process_1.png)

### directory structure

```python
├── AKSVD_function.py     # model and data processing
├── AKSVD_training.py	  # main training code
├── assets
│   └── process_1.png
├── cbam.py               # attention module
├── load_model.py         # main testing code
├── README.md
├── requirements.txt
├── gray                  # BSDS
│   ├── *.jpg
├── Set12                
│   ├── *.jpg
├── test_gray.txt
├── test_set12.txt
├── train_gray.txt
└── visualization.py
```

### Quick Start

1. **Installation**

   ```python
   pip install -r requirements.txt
   ```

2. **Run the training code**

   ```python
   python AKSVD_training.py
   ```

3. **Run the testing code**

   -  **First, set your model path**

     ```python
     # load_model.py
     model.load_state_dict(torch.load("../model.pth", map_location="cpu"))  
     model.to(device)
     model.eval()
     ```

   - **Second, set the name of dataset**

     ```python
     # Test image names:
     file_test = open("test_set12.txt", "r")  # line 57
     
     # Test Dataset:
     my_Data_test = AKSVD_function.FullImagesDataset(
         root_dir="Set12", image_names=onlyfiles_test, sigma=sigma, transform=data_transform
     )  # line 79
     ```

   - **Finally, run the testing code**

     ```python
     python load_model.py
     ```

### Reference

- https://github.com/meyerscetbon/Deep-K-SVD