hCRF is a shared library (output is a static lib) project. This
project is dependent on 5 external libraries, namely asa, cgDescent,
liblbfgs, owlqn and uncOpti.

The library has the following problems:

1. DataSet and DataSequence: 

The destructor of DataSet may not be called if the pointer to dataset
is downcasted to vector as vector has a non-virtual destructor.

Leave-one out and n-fold validation could be implemented using
iterators on the linked list at a very low cost and without the need
to create several copy of the data.

std::vector is not best data structure for this use, every insert is
O(n). Better would be a linked list. 

2. InferenceEngineBP:

The graph should be encoded using a data structure adequate for sparse
graph. I think that adjacency list are adequate, they also offer with
O(1) query time for neighbour if the number of neighbourg is
bounded. I think that boost as a nice graph library.

Have a nice graph member would also reduce the number of class variables.

3.Pointers

The huge number of class with pointers complicate all the copy
constructor and operator=.

4. Toolbox 

A copy constructor and operator= should be defined for toolbox. The
current behavior is a shallow copy, which can lead to double
destruction if a toolbox is copied. The other solution is to have
Toolbox include model and not pointer to model etc.


DebugLevel:
 1-> Basic info about iteration of the optimizer
 2-> Add info about the different gradient and error computation 
 3-> More info about the optimisation (x and gradient)
 4-> Full info 
There is still work to do in some of the Gradient Class.

Running time Optimisation: Gradient is now return the value of the
function. In the case of graphical model, this is free and we gain a
speed up of almost 2. This is due to the fact that the message passing
is the most expensive operation in the training phase.








