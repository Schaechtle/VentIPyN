
Initialise an Experiment
The heart of each experiment is the experiment.ini file. Models, inference routines, specs for testing-data (noise, n of training data), test-problems, mcmc steps, number of repetitions of experiment and the number of points used to compute the residuals are  in experiment.ini:

###############################################################################
#####################    Example experiment.ini ###############################
###############################################################################
[inference]
venture-gp-se=(mh (quote parameter) one 1):1
venture-gp-lin=(mh (quote parameter) one 1):1
venture-gp-per=(mh (quote parameter) one 1):1
venture-gp-lin-p-per=(mh (quote parameter) one 1):1
venture-gp-lin-t-per=(mh (quote parameter) one 1):1
venture-cov-learning=(mh (quote parameter) one 1):10,(mh (quote grammar) one 1):1

[test-data]
data-points= 30,50,100,200
observation_noise= 0.1,0.7
test-problems=function-linear,function-per,function-linplusper,function-linxper

[MCMC]
total-steps-outer = 2000

[others]
repeat=20
number-test-points=10
###############################################################################
###############################################################################
###############################################################################


Inference

All models that are evaluated in an experiment need to be opted in under inference. They are assigned one or more inference routines that are tailored to the model.

The inference string is set in the inference section for each model after the =. The string needs to be a valid inference string for Venture or a stub indicating handwritten inference for cases where Venture is not used.

Every inference string is only one iteration (so that we can collect log-likelihood and residuals with it without having to rely on Venture's collecting mechanisms). We can deal with with one level (as in depth) of nesting at the moment as in:
[(REPEAT 20 (DO (MH (quote outer_scope) all 1) (MH (quote inner_scope) all 10) ) )].

For running inference as above for a model m, just add the following lines to the .ini file:
m = (mh (quote outer_scope) all 1):1,(mh (quote inner_scope) all 1):10
The total number of steps of the outer loop is  added to the MCMC section:
totoal-steps-out = 20

The number of  MCMC-steps in the inner loop is controlled by what comes after the colon. The number of outer MCMC-steps is controlled by the total-steps-outer in the [MCMC] section in experiment.ini. ',' indicate inference steps for different blogs. 

Every single model must have at least one experimental setting for inference. Testing more than one can be  indicated with ';'after each inference routine of interest. 

Models

Pre-defined models are registered in all_registered_models.py and must of the type “Model” which has at least a method to run the experiments:
df=model.run(x_training,y_training,x_test,y_test,f_test_interpolation,f_test_extrapolation,f_error,infer,outer_steps).

In this way, we can register basically any model we want while still controlling what we do during experiments with in experiment.ini.

Venture based models are of subclass Venture_GP_model. By inheriting from this class, one only has to define covariance functions, to test different models.

Run Experiments over conditions:

Every experimental condition of the experiment.ini file will be added to a list to be subsequently run with multi-process ( I guess that there are way more elegant solutions, yet mine is simple and easy to modify)

-----------------------------------------
Problems:
We register problems with synthetic data in the simplest form in all_registered_problems.py (see this file for examples). Real world data set problems need to start with the string “real_world_” in the .ini file. Also, we do not need as much information in the .ini file - we do not rely on noise levels any more and want to define n only in case we want to work on a
subset of the data.
run_experiment sets the repeat index to the number of datapoints in the real world experiments. The repeat index serves in this setting as the index of the left out data point in the cross validation. This index is passed to the experiment function. Where, unlike for the synthetic data sets, we don't call a data generation algorithm but simply load the data,  set x_test,y_test and f_test to the data-point with the bespoken index, set x_train, and y_train to the rest of the data and f_error to zero, since we don't know the baseline. The rest will work as with synthetic data.
ToDo: From now on, we are also recording an interpretation of structure if this is defined in the model. This means, an interp SP has to be assumed in the make gp functions in the individual models.This sp is recorded every sweep. The flag set to record interpretations is set in the .ini file.
-----------------------------------------

Data-Management:

Most of the stuff we need to know is already in the in the experiment.ini file. We are furthermore saving all results in a chronologically named directory (e.g. 2015-05-18, named automatically) as a Pandas-df with the experimental condition in the name so that we can later on use the experiment.ini file to read stuff into a program that does visualisation. The df contains a record of the evolution of the log-likelihood, and residuals for interpolation and extrapolation.

Experiments are never overwritten - if an experiment exists already the programs tries the next one. In case the program crashes during a long run one can simply restart it. Plus, we can set a flag with a command line option indicating a date. If we want to add a condition to yesterdays experiment, all we need to do is set the flag and add the condition to the experiment.ini. 

We are also automatically saving information to a meta file (meta-file.txt), such as the date and time an experiment is run, the experiment.ini file used, and a short summary that we can pass via the command line when the experiment is started. 

The training and test data are saved in a subdirectory of the chronologically named file. Note that we use git on everything but the actual data so that we can later on reproduce experiments.

-----------------------------------------

Running from Command Line

Running an Experiment with command-line options:

run_experiment.py -f experiment.ini -d 2015-05-21 --cores 60 -m “a short message summarising what the experiment is doing and how it differed from the previous one”

--cores is the number of workers used, default is 60, alternatively, by setting the  --local flag, one can switch off the parallelism.  

-----------------------------------------

Investigating the posterior on structure:

We are evaluating the posterior on kernel structure in two different ways:
1. We get the posterior of our model for the airline data and compute a leave one cross-validation of this model compared to what is reported in Lloyd et al., which we train in matlab.
1. For the posterior over structure, we compute a leave on out run with running collect interpretation . We repeat 10 times.
2. We trick Lloyd et al., with a structure that is produced by two latent Gps, show their output on structure, compare it with ours, modelled with an if.

The experiment function will check the model-name (as specified in .ini) for the string “airline” and “trick”, to later on invoke cross-validation schemes and collecting a structure posterior.

-----------------------------------------

Visualisation Tool

We use ipython to plot the results. We can use/change the experiment.ini files for plotting for changing the plots. The function load_experiments(ini_file_path,date) loads experiments into one pandas df and averages results over repeated experiments. Using pandas facet grid, we then plot and compare certain conditions for the evolution of residuals (in terms of RMSE for both inter- and extrapolation) and log-likelihood scores. This can be done with plot_results.ipynb. 

We also allow to just compare the residuals of the last n steps after a fixed number of runs with the function get_last_n_residuals(ini_file_path,date_exp,n_last_residuals) to get an indication of how good the learned function explains the data. This can be done with plot_final_residuals.ipnb.

Edit: I ran a large set of experiments over the weekend. When I played with the IPython notebooks I have realised that it crashes due to DFs that being too large. I am creating a version now that produces graphs and saves pictures on probcomp-3 directly.
