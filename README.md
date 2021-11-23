# Heart MRI Image Segmentation
![alt text](https://github.com/samadon1/heartmri/blob/main/image.png)
## Background
The aim of this project is to build a predictive model that could classify each pixel in a cardiac MRI image based on whether the pixel is a part of the left ventricle (LV) or not.
In medical imagery analysis it is often important to separate the pixels corresponding to different types of tissue, blood or abnormal cells so that we can isolate a particular organ. In this project I used the TensorFlow machine learning framework to train and evaluate an image segmentation network using a medical imagery dataset.
 <br />  <br />
 A working app of the project can be found [here](https://share.streamlit.io/samadon1/heartmri/main/app.py)
 ## Setup
 To run the project locally on your machine <br />
 1. Clone this repository using `git clone https://github.com/samadon1/heartmri.git`
 2. `cd` into the directory
 3. Create a virtual environment and install the dependencies used for the project by using the `requirements.txt` file using the command `pip install requirements.txt `
 4. Finally, start the streamlit app using the command `streamlit run app.py`

## Important
A validation/test file in TFRecord format is already provided for inference.
To make inference with your own file, make sure the file is located in the project directory

## Conclusion
The model for this project was built using GPU and Tensorflow's `tf.data` pipeline. Working with these can be great so moving forward, I'll be creating custom functions to handle imput data provided in other formats aside TFRecords like .png, jpeg etc. Watch this space.
