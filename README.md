# PINN-heat-equation
A Physics Informed Neural Network to solve the heat equation in 1 dimensions.

This project is inspired by Raissi, Perdikaris, and Karniadakis's 2019 paper, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, published in the Journal of Computational Physics (Vol. 378, pp. 686–707). This project demonstrates how a neural network can combine data-driven inference with the governing physics of the system to produce optimized solutions.

## Project Steps
* Construct the heat equation data and adapt it for a suitable neural network
* Create a very basic neural network to compare with the final PINN
* Enhance the neural network by incorporating a physics-informed loss function along with initial and boundary conditions.
* Visualise and compare the effectiveness of the PINN with the basic network

## The Heat Equation
The heat equation is an example of a linear, homogeneous partial differential equation. It is usually written in the form:

$$\Large \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

By using separation of variables, we can obtain a **specific** solution of the form:

$$\Large u(x,t) = \cos\left(\frac{2\pi x}{L}\right) e^{-\alpha \left(\frac{2\pi}{L}\right)^2 t}$$

<img width="702" height="559" alt="image" src="https://github.com/user-attachments/assets/24b6449f-3e2d-48bd-977e-570e54dfeccd" />

Figure 1: The **True** solution of the heat equation plotted

## The Basic Neural Network
A neural network is a computational model composed of layers of interconnected nodes, or “neurons,” which process input data by applying weights, biases, and nonlinear activation functions. During training, these parameters are adjusted using gradient descent to minimize the difference between the network’s predictions and the true solution. This iterative optimization relies heavily on linear algebra operations and can be broadly divided into two main processes: Forward Propagation, where inputs are passed through the network to generate outputs, and Backward Propagation, where gradients are computed to update the network parameters.

**Forward Propagation**

$$\Large z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$

$$\Large a^{(l)} = \sigma(z^{(l)})$$

**Backward Propagation**

$$\Large \delta^{(L)} = (a^{(L)} - y) \odot \sigma'(z^{(L)})$$

$$\Large \delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

$$\Large \frac{\partial J}{\partial W^{(l)}} = \frac{1}{m} \delta^{(l)} (a^{(l-1)})^T$$

$$\Large \frac{\partial J}{\partial b^{(l)}} = \frac{1}{m} \sum_{i=1}^{m} \delta^{(l)(i)}$$

Where: $$W$$ = weights,  $$b$$ = biases,  $$a$$ = activations,  $$z$$ = pre-activations,  $$δ$$ = errors,  $$η$$ = learning rate,  $$σ$$ = activation function

A simple neural network struggles to capture the dynamics of the heat equation—the output barely resembles the true solution, as illustrated below:

![heat_eqn_prediction_vs_true](https://github.com/user-attachments/assets/2f2f0736-a688-4966-998f-97316501c014)

## Building the PINN
To penalise the incorrect solution displayed in the previous network, we can implement 4 different loss functions functions.
* Regular Data Loss
* Physics Data Loss (this is what makes the network a PINN)
* Initial conditions
* Boundary conditions

The final two conditions play a crucial role in ensuring that the network does not collapse to the trivial null solution, which would satisfy the loss function without capturing the true dynamics. To handle these dynamics, the physics loss function encourages the network to output a solution that is mathematically accurate. The implementation of this function is considerably more challenging because it requires the design of a new forward propagation function capable of maintaining a record of the partial derivatives.

<img width="1710" height="580" alt="image" src="https://github.com/user-attachments/assets/a1583e4e-0fe2-48c0-bcb1-b5d628fae3fb" />

<br>

Gradient descent also becomes more complex as the different loss components may conflict with each other. By introducing weighting parameters $$(λ)$$, it allow us to control the relative importance of each constraint, determining how strictly we enforce data fitting versus physics compliance.

<br>

$$\Large L_{\text{total}} = L_{\text{data}} + \lambda_{\text{PDE}} L_{\text{PDE}} + \lambda_{\text{IC}} L_{\text{IC}} + \lambda_{\text{BC}} L_{\text{BC}}$$

These λ values can be tuned in the PINN.py code. I recommend increasing λ_PDE if the solution exhibits massive scaling issues, and increasing λ_IC if the solution remains stuck near zero.
Note that training takes approximately 30 minutes with the default parameters, this could increase with altered parameters. The default solution is shown below:

![pinn_vs_true](https://github.com/user-attachments/assets/5eed4f5a-228e-43e5-b158-1a93ca77ba4b)


