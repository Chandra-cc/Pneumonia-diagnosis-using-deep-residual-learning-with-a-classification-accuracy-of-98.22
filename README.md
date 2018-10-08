# Noval-method-for-pneumonia-detection-using-deep-residual-learning-along-with-separable-CNN
Pneumonia is an infectious disease that inflames the air sacs in the lungs. It may occur in one or both the lungs. The air sacs may fill with fluid or pus, causing cough followed by phlegm or pus, fever, chills and difficulty in breathing. The causes of pneumonia can be a variety of organisms such as bacteria, viruses and fungi. A person suffering from pneumonia apart from suffering from difficulty in respiration, can also suffer from other complications such as bacteremia, lung abscess, pleural effusion and among countless others. 
This model presents a efficient and enhanced method for diagnosis of pnuemonia using deep residual learning along with seperable convolutional networks. 
The images were CXR (Chest-X-ray images) in grayscale format obtained from the mendeley CXR data available at kaggle [Dataset Link](https://www.kaggle.com/parthachakraborty/pneumonia-chest-x-ray) 
The model developed using deep residual learning along with separable convolutional network is given below: 
![Model](freeze_model.png) 
This model was developed on a sample of 3000 images from the mendeley dataset which contained 5857 total images in grayscale format.
This proposed methodology gave an astonishing accuracy of 98.22 % along with an AUROC score of 99.515 resulting in it being one of the most efficient systems for diagnosis of pneumonia. 
# The Preprocessing 
The smapled dataset over which the model was trained contained 3000 images. The data or the images were read in BGR format allowing the learning of the model to be performed to 3 channels. Efficient edge preservation and enhancement techniques were applied including contrast adaptive histogram equalization(CLAHE) and bilateral filtering for precise preservation of the affected lesion in the CXR images. 
![Preprocessing](Capture.PNG) 
# The Model training and the proposed methodology 
The model is based on deep residual learning which reduces the complexity of the network to a much greater extent rather than normal sequential models. Moreover in case where the model gets too deep, residual learning provides better results incrementing the prediction ability of the model. The use of seperable convolutionalk network instead of simple convolutional network is to reduce the network complexity in terms of trainable parameters, time complexity and space complexity. 
The total parameters in this proposed methodology is 157,010 out of which 154,866 are trainable parameters. 
![](Capture1.png)

