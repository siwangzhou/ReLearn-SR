# ReLearn-SR

Run the script 'codes/test_relearn.py' to generate a new optimized LR image.

## Usage Instructions

1.  **Set Up the Environment**
  
    Ensure the required Python libraries are installed, such as `torch`, `lpips`, `pandas`, `openpyxl`, `tqdm` , etc.

2.  **Run the Script**
   
    Run the script `codes/test_relearn.py`. By default, it loads `options/test/test_P2P_HCD_CARN_conv_4X.yml` as the configuration file.
    ```bash
    python codes/test_relearn.py --opt path/to/your/config.yml
    ```
    replace `path/to/your/config.yml` as your actual configuation file path.

3.  **View the results**
   
    The generated optimized LR images and Excel files will be saved in `results/<test_set_name>/<image_name>/` and `results/<test_set_name>/` .
    

