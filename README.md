This project "Brain Tumor Detection and Classification using SVM" is my project that I've done for Master's Thesis.

Abstract:

When we talk about cancer, the most dangerous cancer that is caused by Malignant brain 
tumors are “Brain Cancers”. It is caused by an uncontrollable growth of a cell in the brain 
tissue. Recent studies on this brain cancer have shown to have a high mortality rate if not 
diagnosed in the early stages when the tumor is first detected. So, the main aim of the 
radiologist and a neurologist is to find the early brain tumor. In order to do that, we need 
systems that can detect the tumor automatically for the diagnosis. The common systems 
that medical professionals use like CAD (Computer Aided Diagnostic), MR (Magnetic 
Resonance) Imaging are shown to have positive outcomes, but accurate tumor detection 
and classification is still very challenging till date. It is due to the fact that the tumor 
comes with different structures, proportions, and positions. 

Existing Machine Learning models for detection and classification have outstanding 
accuracy scores. For example, classification models like NasNet, ResNet, and 
InceptionV3 have accuracy scores of 99.6%, 99.7%, and 97.66%, respectively. But when 
it comes to the standard classification model like SVM has a very low accuracy of 91% 
compared to other models. This is due to the fact that it includes drawbacks such as no 
proper parameter tuning, restricted focus on Data Augmentation, and susceptibility to 
technological problems. 

In this project, we are going to look into an SVM model that can detect the tumor and 
then classify according to its properties into 4 classes which are Glioma, Meningioma, 
Pituitary and No-Tumor with a higher accuracy score. Then compare them to other 
classification models like Naïve Bayes, KNN and Random Forest to see if the SVM 
model can be a reliable method for classification.



Results:

Before training the model without PCA and Hyperparameter tuning, we achieved the 
accuracy of 91% and the run time it took just to run the SVM model was around 29 
minutes. And the confidence scores for checking best results for each tumor type was very 
low.

After performing data augmentation in the existing dataset, we created an extra 1200 
images for better analysis and the model underwent PCA and hyperparameter tuning in 
which we have changed the necessary parameters such as C, Gamma, Kernel and Degree. 
Changing these have had a major impact on the SVM model bumping the accuracy to 
outstanding 96.69%. 




Brain tumors - whether benign or malignant - present serious health concerns and 
need diagnosis in order to be effectively treated. The accuracy of tumor identification could be 
greatly increased by using modern machine learning techniques, especially deep learning 
models like Convolutional Neural Networks (CNNs). However, there are still issues with 
correctly categorizing tumor types because of the limitations of some models, such as Support 
Vector Machines (SVMs), which can have issues with performance and accuracy. 
In order to overcome these obstacles, model performance can be greatly improved by using 
sophisticated algorithms, adjusting hyperparameters, and using data augmentation strategies. 
Other state-of-the-art models, including the Random Forest Classifier, have shown greater 
accuracy of 97.8% when trained on the same enhanced dataset, whereas the SVM model only 
reaches 96% accuracy. 
We can attempt to create more thorough and precise classification systems by utilizing these 
developments. This development is crucial for increasing the accuracy of diagnoses and 
enabling rapid, focused treatment regimens for brain tumor patients. 











