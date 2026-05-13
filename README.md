# Dynamic-neural-engine
# Introduction
* This is a dynamic neural engine which can create however many hidden layers in the network the user wants.
* This project only uses numpy for better understanding of how the model can create n number of layers.
* The model also uses Nestorov Accelarated Gradient to make the model converge faster.

# Working Principle
* The model uses basic for loops to create the neural layers and appends them in a list so that we can use them for the forward pass and backpropagation.
* It uses NAGs(Nestorov Accelarated Gradient) for faster convergence which is a better version of momentum.
* In momentum we update the velocity each iteration which builts up over the course of the training process and we use the velocity to update the weights and biases. But the momentum can sometimes make the gradients overshoot and it has to find the global minima by going in the opposite direction.
  
                          v = gradient*learning rate + v*g
                          g is the percentage of the previous velocity we want to add in our current step
                          
* NAGs eliminate this by looking ahead of the slope before updating the weights. Before training the weights are updated by the velocity so that the they can look ahead of the slope this eliminates the problem of
  over shooting and helps the model converge even faster
* if SGD is a person walking down the slope and momentum is a bowling ball which is rolling over building momentum and escaping local minimas using the momentum, the NAG is a smart ball which corrects it's speed
  based on the terrain before it.

# Goal
* To show that more number of hidden layers does not always mean faster convergence.
* To show the power of NAGs
* To find what is the ideal number of hidden layers for learning Full adder in 5000 iterations.

# Dataset and Functions
* We are going to make our model learn a Full adder.
* This is a classification probem so we use both leaky relu and the sigmoid function.
* The relu function

                    f(x):{x>0,x
                          x<0,0.01*x}
                    f'(x):{x>0,1
                           x<0,0.01}
* The sigmoid function

                    f(x): 1/1+ e power(-x)
                    f'(x): x(1-x)
# How To setup and run
🚀 How to Run
Follow these steps to set up the environment and run the dynamic neural engine on your local machine.

1. Clone the Repository
Open your terminal and run the following command to download the project:


* git clone https://github.com/DanielDavid-H/dynamic-neural-engine.git
* cd dynamic-neural-engine

2. Install Dependencies
This project requires NumPy for matrix multiplication and Matplotlib for visualizing the training loss.


* pip install numpy matplotlib

3. Execute the Model

Run the main script to start the training process.

python dynamic engine.py

4. Training & Prediction

Step A: The model will ask number of hidden layers.

Step B: The model will train for 10,000 iterations using backpropagation.

Step C: The model will print the loss values and a Loss Curve window will pop up.




