# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="828" height="830" alt="image" src="https://github.com/user-attachments/assets/e0fc9f5d-f848-48b0-9b24-f5dc9848785d" />


## DESIGN STEPS
### STEP 1: 
Load the customer data, then clean it by handling missing values and encoding categorical features into numerical format.

### STEP 2: 
Divide the data into training and testing sets, then scale numerical features to standardize their range.

### STEP 3: 
Define a multi-layered neural network model using PyTorch to learn complex patterns for customer segmentation.

### STEP 4: 
Train the neural network model on the training data, optimizing its parameters to minimize the prediction error.

### STEP 5: 
Assess the trained model's performance on the test data using metrics like accuracy, confusion matrix, and classification report.

### STEP 6: 
Demonstrate the model's ability to classify a new, unseen customer into one of the learned segments.

## PROGRAM

### Name: Malar Mariam S

### Register Number: 212223230118

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)


    def forward(self, x):
      #Include your code here
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x;
```

### Dataset Information
<img width="998" height="736" alt="image" src="https://github.com/user-attachments/assets/bcf23d46-d51c-4154-8b8f-d0efdad4bae0" />

### OUTPUT

## Confusion Matrix
<img width="539" height="455" alt="image" src="https://github.com/user-attachments/assets/edaa63db-ac94-49ff-8c45-b0eab36d524d" />

## Classification Report
<img width="745" height="257" alt="image" src="https://github.com/user-attachments/assets/3aa15c1d-fbc7-4d79-9334-a5b499a6d110" />

### New Sample Data Prediction
<img width="716" height="100" alt="image" src="https://github.com/user-attachments/assets/f6134810-25d4-4661-b3ac-68864372c419" />

## RESULT
A neural network classification model was successfully developed and tested on the given dataset with satisfactory classification performance.
