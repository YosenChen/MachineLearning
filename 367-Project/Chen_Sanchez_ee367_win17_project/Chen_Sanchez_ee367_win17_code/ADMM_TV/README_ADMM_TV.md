ADMM_TV:
All necessary data and functions are provided to run the following
scripts. 

To generate ADMM_TV results for optimal lambda that minimizes the MSE:
run Hyper_param.m 
NOTE: Hyper_param.m is a script, to find the optimal lambda, it takes 
      approx 6 hours to run current setting for 5 lambdas and 6 images! 
      To skip this part, run the first and last sections of this script 
      with a given lambda value. To generate the actual results in the 
      report, you will need to let this entire script run with all its 
      sections! 


To generate optimal lambda for a set of 3 images:
run DMD_opt.m
NOTE: DMD_opt.m is a script, this make take approximately two hours to run.


To generate the Blinear and Malvar results:
run Demo_methods.m
NOTE: Demo_methods.m, is a script, this should take a few seconds to run.