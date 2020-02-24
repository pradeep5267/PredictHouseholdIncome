NOTE:
- LINUX DISTRO UBUNTU WAS USED IN THE ENTIRE PROCESS
- The trained model is resnetV2 152 although trained on 
google colab using GPU, the inferrence is done on CPU to avoid compatibility issues which
occurs due to different tensorflow gpu environments and CuDnn versions.

- Why resnetV2 was chosen:
    I experimented with VGG16, VGG19, ensemble of 9 CNN, ensemble performed better with an accuracy
    of 41 percent when compared to VGG16, VGG19 with 32, 37 percent respectively.
    resnetV2 although a bigger model than VGG16 takes slightly less time to train and infer upon, 347s for 
    resnetV2 when compared to 359s to VGG16 on google colab.
    resnetV2 gave an accuracy of 83 percent with 30 epochs and inorder to reduce overfitting the 
    trained model was retrained (or knowledge transferred) using the same training set but with data augmentation
    which makes it more robust to noise and less dependence of high level features since a multiclass classifier
    has to rely on both pixel level features and high level features.

- WORKING
- the python script is named as PredictHouseholdIncome_pradeep4444.py
- At line 114,115,116 change the input_path to test folder containing
test images, label_path to the folder containing labels in excel file and the target column name
(the target column name is set to label by default)
- function output_make at line 92 is the main function which calls other
helper functions
- the make_label array is reads the excel file from label_path and creates a
dataframe with one column by reading the target parameter passed to the function.
- It then converts it to a numpy array and one hot encodes it using np_utils.to_categorical.
- the test images are read and converted to gray scale, then a 3rd dimension is added to make it compatible with resnetv2, its then
stored as a pickle file, and then the pickle file is read and normalized (divided by 255).


- Create the required environment using the provided env.yml file or 
requirements.txt file; it is preferable to use yml file since some packages are
installed using conda and some using pip

