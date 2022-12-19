
# TEN YEAR RISK OF FUTURE CORONARY HEART DISEASE PREDICTION USING MACHINE LEARNING 


In this project, we develop a 10-year risk of future Coronary Heart Disease Predict (CHDP)system that can assist medical professionals in predicting CHD status based on the clinical data of patients. 
It is an developed version of Heart Disease Prediction System (HDPS). Our approaches include three steps. Firstly, we select 15 important clinical features. Secondly, we apply Machine Learning algorithm for classifying CHD based on clinical features. The accuracy of prediction is nearly 83%. Finally, we develop a user-friendly 10-year risk of future CHD system. 


## EXISTING SYSTEM

In the past few years, a lot of projects related to a heart disease prediction have been developed. Work carried out by various researchers in the field of medical diagnosis. 
Diagnosis of the condition solely depends upon the Doctor’s intuition and patients records. The major drawbacks of the existing system are: 
- Detection is not possible at an earlier stage. 
- The model is trained on very few data points.

## PROPOSED SYSTEM
The proposed system predicts the 10-year risk of developing Coronary Heart Disease by using a machine learning (KNN) algorithm. 
The advantages of proposed system 
  - Increased accuracy for effective heart disease diagnosis.
  - Handles roughest(enormous) amount of data using KNN algorithm. 
  - Reduce the time complexity of doctors.
  - Cost effective for patients.

## SYSTEM REQUIREMENT SPECIFICATIONS
SOFTWARE REQUIREMENTS: 
- PYTHON
- KNN ALGORITHM
- WINDOWS 10
- DJANGO
- AWS S3
HARDWARE REQUIREMENTS:
- PROCESSOR  : Core i7
- HARDISK    :  512 GB
- RAM 		 :  8 GB
## DESIGN

K-Nearest Neighbour (KNN) ALGORITHM: 
K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.

Example: Suppose, we have an image of a creature that looks similar to cat and dog, but we want to know either it is a cat or dog. So for this identification, we can use the KNN algorithm, as it works on a similarity measure. Our KNN model will find the similar features of the new data set to the cats and dogs images and based on the most similar features it will put it in either cat or dog category.

## ARCHITECTURE OF CHDP
![architecture](https://github.com/munazzaznoor/project_heart_disease/blob/master/screenshot/architecture.png?raw=true)


![architecture](https://github.com/munazzaznoor/project_heart_disease/blob/master/screenshot/predictions_during_devlopment.png?raw=true)


![user_interfeace](https://github.com/munazzaznoor/project_heart_disease/blob/master/screenshot/user_interfeace.png?raw=true)
![making_predictions](https://github.com/munazzaznoor/project_heart_disease/blob/master/screenshot/making_predictions.png?raw=true)
![result](https://github.com/munazzaznoor/project_heart_disease/blob/master/screenshot/result.png?raw=true)


## FUTURE ENHANCEMENT
- At some point in future, the machine learning model will make use of a larger training dataset, possibly more than a million different data points maintained in the electronic health record system. 
- Although it would be a huge leap in terms of computational power and software sophistication, a system that will work on artificial intelligence might allow the medical practitioner to decide the best suited treatment for the concerned patient as soon as possible. 
- A software API can be developed to enable health websites and apps to provide access to the patients free of cost. The probability prediction would be performed with zero or virtually no delay in processing .

## REFERENCE

- Introduction to machine learning with Python by  Andreas C Mueller, Sarah Guido.
- Python web development with django by jeff Forcier, ‎Paul Bissex ,‎ Wesley J Chun.
- https://www.kaggle.com/amayomordecai/heart-disease-risk-prediction-machine-learning
- https://medium.com/analytics-vidhya/machine-learning-project-11-whose-my-neighbor-k-nearest-neighbor-3e9184ce5f89
