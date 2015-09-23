#Types of Machine Learning

1. Supervised 
2. Unsupervised
3. Hybrid-RL

Perceptron fits under supervised.

#Perceptron


![alt text](https://dwave.files.wordpress.com/2011/05/qc_ai_diag1b.jpg "Picture of peceptron")

The Perceptron is made up of 3 main components:
Input x<sup>i</sup> (can be many dimensions, 2D - yes/no, higher - what's in a picture? dog, cat? etc.) 

Output: ŷ (what the machine spits out)

Correct: y<sup>(i)</sup> (what the machine should say)

Dials/knobs - weights <sup>(i)</sup>,  w is a vector of numbers usually but could be a data structure, or many seperate vectors but we'll treat it as one 

You can turn and fiddle with the dials to make things nice, usually by some objective function.
Try hard to get the correct answer - non-objective, this is what it used to be, they used rule of thumb.

Perceptron is an early ML alogorithm which fiddles with the dials to get the output correct for a given input.
If you do this repeatedly for a data set you can prove then it will get the correct answer on all of them.

Usually we have some error measure, usually a fraction of what it gets right, we'll call this E, we want this to be low.

E = error over entire training set

E = 1/m(∑(i=0, m-1) E<sup>(i)</sup>)  sum over all cases or average, where m is the number of inputs
  = 1/m( ∑(i) E<sup>(i)</sup>)
  = <∑<sup>(i)</sup>>    average over i

#Construction of Perceptron

Outputs are binary; 3 options: **True/False**, **1/0**, **+1/-1** 

1/0 useful if you expect lots of zeros eg. hand-written digit recognition

+1/-1 useful if you're expecting some sort of symmetry and can use this to figure out if it was right or wrong

w<sup>(j)</sup> will be  our voltage and θ is our threshold so we end up with equation:
ŷ = ∑(j) x<sup>(j)</sup> * w<sup>(j)</sup> > θ

Basically the inputs x<sup>(j)</sup> are sent in and run through wires going through resisitor which are controlled by the dials, these are then added up and the machine then checks if this is greater than the threshold.

Not quite what we want, we want them all to be a whole function

so ŷ = sign(∑(j=1, n) w<sup>(j)</sup>*x<sup>(j)</sup> - θ) this is the Transfer function of a Perceptron. This equation will return -1 if ŷ is less than the threshold and +1 if it's greater than it. 

So what was the motivation behind the Perceptron?
# Neurons

![alt text](http://www.explorecuriocity.org/Portals/2/article%20images/3756/1280px-Neuron.svg.png "Neuron diagram")

In the late 1940’s, it was figured out how neurons work. Alan Hodgkin and Andrew Huxley performed experiments on the giant squid axon, recording ionic currents (for which they got the 1963 Nobel Prize in Physiology and Medicine)


The potential difference between the inside of a neuron and the outside is -80mV. A pulse is transmitted through the axon terminal, causing a chemical reaction and the “gate opens up”. This allows some particles through - ions. Sodium rushes into the neuron, and the voltage increases past the threshold - this is a spike. Now other neurons can figure out what the neuron is doing.

Neurons have "amplifiers" on them called axons, this is because if the signal was too small it would die out and if it was too big it would keep amplifying. 

In a neuron, the maximum spiking rate is 1kHz. This rate is only seen in a dieing neuron though. The average spiking rate is ~0.1Hz.

Why don’t we use chemical reactions like this in computers? Cause its slow!


## How to adjust the weights on the machine?

We can change 2 things, the weight or the threshold. 
For our example we'll set theata = -w<sup>(0)</sup> and x<sup>(0)</sup> = 1

ŷ = sign(∑(j=0,n) w<sup>(i)</sup>*x<sup>(i)</sup>)  (we'll call the bit in brackets z ie. z = ∑(j=0,n) w<sup>(j)</sup>*x^<sup>(j)</sup>)

Imagine we have a LEARN button on the machine, if this is pressed it can do 2 things:
If ŷ and y are the same, do nothing. Otherwise adjust the weigths accordingly.

| ŷ      | y          | do |
| ------------- |:-------------:| -----:|
| +1     | +1 | nothing |
| -1   | -1   |   nothing  |
| -1 | +1      | change w to make z > 0 |
| +1 | -1      | change w to make z < 0 |

In modern techniques you make very small changes to the weights over and over, until you get the desired output. With the Perceptron it would make a big adjustment, so that it got the right answer, but the smallest change possible to get this result.

But how can we do this? Use Linear Algebra. Think of it as symetrically pushing vectors around.

![alt text](http://www.willamette.edu/~gorr/classes/cs449/Classification/perceptPict-2.gif)

Input space - x ϵ R<sup>(n)</sup>  (this is the space the inputs of the machine can "live" in)

In our case it will be a 2D input but it could be more (video, text, audio etc.)

Aside - ours is actually 3 as we set xo = 1, but we can ignore this for maths purposes

Weight space - w  ϵ R<sup>(n)</sup>

Output space - binary (yes/no or in our case +1 or -1)

How are we going to change it so that we get the correct response? We've 2 options of what to look at:

For a particular w we look at the input (x), break it up to which we give our to plus or minus 1.

For a particular x we look at weights (w) and see which give plus or minus 1.

In this case w is a seperating surface, and we need to turn w. In our graph we'd want to turn it clockwise, but what about in general? It's not always 2D.

We want to move w towards x. 

| ŷ      | y          | do |
| ------------- |:-------------:| -----:|
| +1     | +1 | nothing |
| -1   | -1   |   nothing  |
| -1 | +1  | change w to make z > 0   w(t+1) = w(t) + δx |
| +1 | -1      | change w to make z < 0 w(t+1) = w(t) - δx |

w̃(t+1) =  w(t) - δ*x

We want to solve: (w + δx) . x = 0 (ie. solve for 0)  (the dots on this and the next few lines represent dot products)
w.x + δ*x.x = 0
w.x + δ||x||^2 = 0
δ = w.x/||x||^2  (this would be just on the seperation plane)
δ = - w.x/||x||^2 + d (the plus d ensures its just over the sepreration plane)


To make it work for both cases:
δ = (-w.x/||x||^2 + d)*y
δ = (-w.x/||x||^2 + d)*(y-ŷ/2)  (this accounts for the do nothing case

What δ? Normailse w? Not much differnce, convinient not too big or small

If the output is wrong, press learn and it'll be right next time.
If it sees all inputs it'll get all of the outputs right.

Aiming for the smallest tilt to get it right.
If theres no setting to get it right then it won't work.


#Notes from previous years

NOTE : Pictures wouldn't cross over so only text

Perceptron Learning Rule

Input comes in
Checks if correctly classified
	if it is → do nothing
if it’s not → change weights by the minimum amount (rotate w a little bit) so that it would be correctly classified

 

Convergence rule: if w exists, then by cycling through the inputs you’ll eventually find w, where all inputs are classified correctly.

w doesn’t exist if there is even a single outlier. No admissible solution.



 Perceptron:
•         Invented in 1957 at the Cornell Aeronautical Laboratory by Frank Rosenblatt.
•         Funded by the United States Office of Naval Research. Used to distinguish tanks from their surrounding environment.
•         This machine was designed for image recognition: it had an array of 400 photocells, randomly connected to the "neurons". Weights were encoded in potentiometers, and weight updates during learning were performed by electric motors.
•         It was later realized that the perceptron was influenced not only by the shapes of images given for its interpretation but it also was effected by the brightness and was unable to clarify the presence of tanks when there was a different brightness to the time the tank present data was taken.
How does the perceptron work?
 
Figure 1. :  is a graphical illustration of a perceptron with inputs , ...,   and output   (sourced from http://reference.wolfram.com/applications/neuralnetworks/NeuralNetworkTheory/2.4.0.html)
As seen in figure 1 the weighted sum of the inputs and the unity bias are first summed and followed by being processed by a step function yielding the output
 	(x, w, b)= UnitStep (w1 x1 + w¬2 x¬2 +  . . . + wn xn + b)
Where {w1. . . wn} are the weights applied to the input vector and b is the bias weight. Each of the weights are represented by the arrows in figure 1. The Unitstep function is 0 for arguments less than 0 and 1 elsewhere. So   can take values of 0 or 1 depending on the value of the weighted sum. The perceptron can indicate 2 classes corresponding to these 2 input values. While in the training process, the weights (inputs and bias) are adjusted so the input data is mapped correctly to one of the two classes.
Off sample performance more important!!!




Cross validation
Cross-validation, sometimes called rotation estimation, is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. Generally better for a scenario where the goals are predicted, and the test is to see how accurate the prediction is by having a training set and a validation set. The goal of cross validation is to define a dataset to "test" the model in the training phase (i.e., the validation dataset), in order to limit problems like overfitting, give an insight on how the model will generalize to an independent data set (i.e., an unknown dataset, for instance from a real problem). 
A bigger data set presumably gives a better generalization!
figure 1: the line between +’s and –‘s is constantly moving until it fits to a particular position where the margin between the –‘s and +’s are equal.
 A problem with this will be if they aren’t separated appropriately. 3 common problems causing this may be outliers, noise and mislabelled points. It is best to keep the margin between +’s and –‘s as large as possible. 
Margin= distance from separating surface to nearest point, assuming points are correct. 
The closest points to the separating surface are known as support vectors and they are used to help calculate the maximum margin. 
 



