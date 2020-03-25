# 07/18/2019 
## 4:00 A.M.
## Refactored Genetic Algorithm Approach

The new, refactored implementation of genetic algorithm provides for a cleaner way for the user to perform the 
operations. However, in the future, it might be necessary to further automate the genetic algorithm loop somehow
by trying to automate
	1) Simulation submission
	2) Incrementing of GENERATION_NUM
Additionally, the ANN must be implemented in the future.

## Regarding Genetic Algorithm Saturation
## Simulation Folders: 20190716-GA-NEW-method and 20190718-GA-NEW-Testing

I have noticed that as the generations go on, the entire population starts converging to a singular total_error.
This must be due to the fact that mutations and crossover are performed on ONLY THE SELECTED (i.e., top 10%) of
children from the previous population. Thus, as time goes on, the fittest survive, and mutation and crossover
does not significantly alter their genes. Crossover should result in zero `diff` between children; however,
mutations should still demonstrate minor differences in the population.

# 7/18/2019
## 4:40 P.M.
## Continued Refactoring of GA Approach

I wanted to be able to generalize the implementation of the cost/fitness function for the genetic algorithm for
the future, in case we wish to check more than just the `total_error` of each child/case. So, I implemented 
kind of a strategy pattern in the Child class (child.py module). Now, the Child @property `cost` can be used, as 
long as it is implemented to return a single floating type value. 

## Regarding the Old Approach
## Simulation Folders: 20190718-GA-ANN-OLD-METHOD and 20190707_GA_ANN_test_1

I believe there may be some errors in the old method (GA + ANN). I tried some test runs...initially, the total error
for the master GA was dropping, but in some cases, the total error from the best case from one generation to the
next generation increased! This could either be due to a bug in the genetic algorithm, or due to poor predictions
from the ANN...

# 7/24/2019
## Started S-Glass Training
## Currently testing GA only
## Simulation Folders: 20190724-S-glass-GA-ONLY-TESTING and 20190724-S-glass-GA-ONLY-TESTING

Finally, we have started training for the S-glass parameter set. The old methodology was producing errors since
it was reading 'fort.99' in the very first generation, but the new, refactored methodology is working fine.
The current problem is that no 'fort.99' files are being generated, I suppose since all ReaxFF optimizations
are crashing.

# 7/25/2019
## Continued S-Glass Training
## Currently testing GA only
## Simulation Folders: 20190725-S-glass-GA-ONLY-TESTING

'fort.99' files are being generated now; looks like the problem was on Joon side of the story. However, the problem
is that for the first trainings, the error is too large, so 'fort.99' generates '*****.****' (large values for error)
and the last two columns (error and accumulated error) merge into one. I refactored the code to calculate the error
myself using the reaxFF error calculation (error = ((reax_pred - true_val) / weight)^2). I will try it out and see
if it works for the second generation.

# 7/26/2019
## Continued S-Glass Training
## Currently testing GA only
## Simulation Folders: 20190726-S-glass-GA-ONLY-TESTING

I have successfully read 'fort.99' output values (first three columns: reax_pred, true_val, and weight) using 
regular expressions! Going to now test the script and see if it works.

# 7/28/2019
## Continued S-Glass Training
## Currently testing GA only
## Simulation Folders: 20190727-S-glass-GA-ONLY-TESTING

I have successfully implemented job scheduling! This should make running several generations of the genetic algorithm
very easy. I am now testing that, as well as working on implementing the artificial neural network, although I need
Jeffrey Frey to install the pandas library for me. Progress should be a lot faster with automated GA though.
However, after several generations, I am running into a problem: the error is converging to a value of 2 million!
I am going to try a larger scaling factor and see if that helps.

## Continued S-Glass Training
## Currently testing GA only
## Simulation Folders: 20190728-S-glass-GA-ONLY-TESTING

Since I was experiencing premature convergence, I decided to experiment with a larger scaling factor for mutation.
But that created a new problem: columns started merging in the 'ffield' file...So, I am going to try again with a 
smaller scale factor (2.5 * param_increment instead of 5.0 * param_increment). Maybe I should work on redoing
how the data is being read for that file, so that no such problems arise.
Even a scale factor of 1.5 eventually lead to errors. Going back to using 1.0 and continuing from generation 3.
Even with a scale factor of 1.0...I got errors...

## Simulation Folders: /lustre/scratch/daksha/20190728-S-glass-GA-ONLY-TESTING-new-reference

Trying now with new trainset.in, params, and ffield files.
New trainset.in, params, and ffield combination was throwing an error.
Now I am trying with just new trainset.in file.

# 7/29/2019
## Continued S-Glass Training
## GA only
## Simulation Folders: 20190728-S-glass-GA-ONLY-TESTING

Looks like 19 generations have gone by, and the error has decreased from ~3.2 million to ~2.2 million. However,
the error is now starting to stagnate. I am going to try starting the generation over, but using reference files
from the latest generation.

# 08/02/2019
## GA + ANN
## Simulation Folders: 20190731-S-glass-GA-and-ANN-TESTING

Collaborated a lot with Jeff and finally got the ANN + Pandas to be working on Caviness. Now it is time to test
some generations with the ANN...it looks like the population was being overfitted. Maybe there is not enough data,
or maybe I have to reduce the number of nodes in the hidden layer.

# 08/16/2019
## GA + ANN
### Simulation Folders: 20190816-S-glass-testing

Talked a lot with Joon...major improvements to the algorithm need to be made. 
	-Need to implement better crossover/mutation algorithms
	-Need to implement adaptive GA
	-Need to figure out issue with weights
	-Need to prevent local convergence
Today, I made it so that the nested GA only runs if the validation r-squared value is more than a certain threshold
(for example, val_r_square > 0.80). 
Also, I need to consider removing duplicate lines from `params` file and see if that changes the results.

# 08/17/2019
## GA + ANN
### Simulation Folders: 20190816-S-glass-testing

For some reason, after some generations of training, the ANN started generating `NaN`...not sure why this is happening.
I will need to debug this. Also, I have decided to currently set the crossover fraction for the nested GA to be equal
to zero. This is to prevent huge changes during the nested GA iterations, since the ANN may be unable to predict
the results for such cases. 
I think operating the GA with a large crossover rate (and thus a low mutation rate) may be better. I should potentially
begin testing with the GA only algorithm and see if that works out.

## GA only
### Simulation Folders: 20190817-S-glass-testing-GA-only

Looks like increasing the crossover rate significantly and using GA only still resulted in convergence of error to
local optimum. I am now going to try deleting duplicates from the params file and see if that does anything.
NEVER MIND, it turns out the lines in the `params` file were already uniques.
I am going to try testing with the ANN once more and see if the `NaN` issue pops up again.
Now, the crossover fraction is zero, so performance may be better.

# 08/18/2019
## GA + ANN
## Simulation Folders: 20190817-S-glass

I am now thinking that I should try the training by setting all weights equal to 1.0--that is, to try and calculate
the errors without a weighting factor. I will do that in the next run and see how it goes.
Another idea I want to try is to use more hidden nodes in the hidden layer of my ANN. Maybe even more hidden layers.
I want to make sure that when a model is being loaded and fitted once more, it is not forgetting what it has 
previously learned. I can maybe test this by trying to train the ANN individually for each generation,
collecting all data from previous generations for the training set. Also, to be consistent in my regularization step,
I should potentially save the training stats--mean, standard deviation--and improve upon it as new data is introduced.

new_avg = (num_old_examples * old_avg + num_new_examples * new_avg) / (num_old_examples + num_new_examples)


# 08/28/2019

I am back at University and I am now ready to conduct my senior thesis. There are some immediate changes that need 
to be done:

-Implemention of better crossover/mutation algorithms
-Improving how the weights are being considered
-+/- sign checking

For now, I am going to be focusing on improving the standalone GA. Then I will take steps to take a look at the ANN.
ALSO, there may be something wrong with only performing mutation and crossover with only the top x%, i.e., the 
selected individuals in the genetic algorithm. I will look into this to see if I should be using more of the 
population for the selection/mutation criteria.

# 08/29/2019
## GA ONLY 
### Simulation Folders: 20190829-SiF-GA-ONLY

I have done a lot of refactoring of the GA crossover and selection, and I am now going to test the new algorithm out 
on the SiF training set!
So far, it looks like the algorithm is working very well! The error is consistently decreasing and not converging
at a high value. Let's see how it keeps going.
On the other hand, I need to implement an easy way to plot all the errors as the generations go on.

I fixed an error in the `trainset.in` file. I hope that didn't cause any errors. It was generating a line in `fort.99`
at the end of the file with an error of NaN and a weight of zero.
I tried seeding to be able to get consistent results later. Big mistake! Seeding the stochastic algorithm created a big
issue and the total error did not decrease for ~70 generations!
Current problem with tournament selection algorithm: it samples from random.sample(range(pop_size)). But sometimes, not
all children in a generation finish! So I need to fix this.
This is fixed -- I am now just using range(len(population)).


# 08/31/2019
## GA ONLY 
### Simulation Folders: 20190829-SiF-GA-ONLY

I need to develop a better mutation algorithm. But, it seems most algorithms already assume that you have [min, max] bounds
for your parameters...I need to work on getting [min, max] bounds for ALL parameters in the `params` file. 
In the meanwhile, I am going to try doing something else instead of Nakata's mutation algorithm:
I am going to select a random fraction of parameters and reset them based on the uniform distribution based on [min, max].

EDIT: Joon sent me a file with [min, max] bounds for all parameters in the `params` file. I want to implement polynomial
mutation.
I have implemented the polynomial mutation algorithm. It is time to test the difference!

Note: I have stopped 20190829-SiF-GA-ONLY runs after 1,025 runs. It was going pretty well, but it seems that the population
stagnated between generation ~950 - 1025. Maybe if I gave it more time it would have improved, but I want to test the 
new mutation algorithm anyways. Also, I need to try testing a different population size.


# 09/01/2019
## GA ONLY
### Simulation Folders: 20190901-SiF-GA-only-improved-mutation

Note: The initialization step (for the first generation) is still being done using Nakata's scaling method.


I have implemented a simple adaptive GA algorithm using Lei and Tingzhi's paper. It introduces two new parameters, but I will
have to test it out! Also, it seems like common mutation rates (according to their paper) are usually around 
~0.005 - 0.1, so I should consider lowering my mutation rate, as it is currently 0.1...maybe I should try 0.01.

Generations are taking forever to complete, probably because Caviness is super busy...
I need to test with a smaller population size and see if I can still achieve decent results.

I have done ~1000 generations with the new mutation algorithm. It looks like the GA is consistently decreasing, but it is still
definitely pretty slow. Maybe I should introduce elitist criteria, so that the best solution is always pushed forward.
Anyways, now I am going to try the adaptive GA approach! After that, I should try reducing population size.
I also need to try smaller starting mutation rate (~0.01 maybe?).


# 09/03/2019
## GA + Adaptive GA
### Simulation Folders: 20190903-SiF-AGA

The GA is performing very well relative to the two previous test runs. It's not because of AGA, since I have been tracking the 
mutation and crossover rates and they have been constant. Instead, I think it is because I changed the mutation eta parameter
(controlling the spread of mutation: how close it is to the parents) from 10 to 5, making the spread larger. I will keep tracking
progress and see what happens.

Decreasing the eta parameter definitely made the algorithm better. For now, I will continue sticking to eta = 5. 
Now, I am going to test decreasing the population size from 100 to 50 and see if the algorithm still has a decent performance.
I also tried changing the `B` parameter in the Lei adaptation method from 0.9 to 0.8, since the AGA never actually took 
place.
Next, I should try decreasing the mutation rate and maybe changing the crossover algorithm. Then, I can see if the ANN has a 
positive influence on the convergence.


# 09/07/2019
## GA + Adaptive GA
### Simulation Folders: 20190907-SiF-AGA

I am going to stop the optimization runs that are testing with population size of 50. I am pretty sure the performance with
a bigger population size was most likely significantly better, so I will focus on testing other variables; if I have time,
then I can do more extensive testing of the population size.

Now, I am going to try changing the crossover algorithm to the uniform crossover. Also, I am reintroducing the 
`JOB_THRESHOLD` variable in the `main.sh` script and setting it equal to 5; Now, if only 5% jobs are remaining, those jobs
will be canceled and the next generation will be started.

I should implement the crossover such that multiple crossovers can be selected. Maybe even the mutations.
I should improve the file structure of the project.
I should possibly introduce elitist criteria so that the best child does not get replaced.
I should possibly make it so that the same parent cannot be chosen for tournament selection.


# 11/07/2019
## GA + Adaptive GA + ANN

I have decided to implement the ANN such that it reads N number of previous generations (currently 10) to fit the data.
Now, I need to try and maximize the accuracy of the ANN and test whether it is actually improving the GA.
I am going to test with Dropout layer (rate = 0.2) for the input layer and with L2 regularization (lambda = 0.1). 
I am also trying to increase the fraction of training set data from 0.8 to 0.9
Relevant simulation folder for this test: 20191107-SiF-improving-ANN-accuracy/

# 11/08/2019

I found the problem with AGA. Liu's method inevitably can lead to probabilities of crossover and mutation that are greater
than one...it wasn't an implementation issue. And it won't really make a difference in the outcome, but I have fixed
the code such that the maximum crossover and mutation rates will now be equal to one.

From the 20191107 test, it seems that even with ANN, the GA is having trouble with premature convergence (around ~130K error),
which is similar to the best error found previously...

Anyways, now, it is time to start systematic benchmarking tests. I am going to run 200 generations with 3 independent samples, 
first comparing the effect of having all weight values equal to one with using Joon's optimized weight values.
Note that I am using GA only for this test.
Relevant folder for this test: 20191108-SiF-weight-value-testing/

Looks like waiting for 1000 generations takes too long. I might need to establish other criteria for stopping, or maybe I should
go until a smaller generation (maybe 500 or even smaller).

I think I need to improve the crossover scheme -> instead of binary, I am thinking of using crossover with N number of cuts.
I am also thinking of incorporating a weighted average of the parents for crossover (and using a mix of both aforementioned 
algorithms). I could do the same with mutation - maybe use a mix.

# 11/19/2019

I am changing mutation rate from 0.20 to 0.10 because I believe 0.20 is too high, especially when the nested GA might start
being applied. Also, I am deciding to stick with usage of one mutation (polynomial) and one crossover (uniform) operation each.

# 01/31/2020

Happy New Year! Some things have changed. Anyways, recently, I have started writing my thesis, and I have encountered a problem
I must address. The crossover operation---which I originally took for granted---obviously plays a large role in arriving at an
optimal solution. Unfortunately, the current implementation (uniform crossover) may have several problems that prevent it from
arriving at the best solution. Moreover, since the crossover algorithm has a large effect on the solution determination, the 
overall GA suffers tremendously if a poor crossover algorithm is used.

I am thinking of trying either SBC (simulated binary crossover) or LX (laplace crossover)...I will decide on one soon and move
forward.

I will also work on changing my mutation algorithm. I have recently been using an adapted version of polynomial mutation that
no longer requires lower and upper bounds, but this may not be a good scheme (also, I am using Nakata's parameter mutation
for the initialization). I am looking into using either Gaussian mutation or Cauchy mutation (I have also been reading about
Levy's mutation scheme), or a combination of the two. I will decide on one soon and move forward with the implementation.
