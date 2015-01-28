Intro to Bayes Short Course
======

**DESCRIPTION:**

This is a short 4-class introduction to Bayesian Statistics. 
We will illustrate basic concepts of Bayesian Inference 
by using mostly Astrophysical examples.

**REQUIREMENTS**

- Python modules required are NUMPY, SCIPY and MATPLOTLIB.
- For lesson 3 we will show an example with the MCMC sampler emcee
  from Foreman-Mackey et al. (2013)

**FILES PROVIDED**

- Executable programs
   * L1_progs/moneda_priors.py
   * L1_progs/moneda_ex.py
- Lecture notes
   * L1_intro_to_bayes_CU.pdf 
   * L2_intro_to_bayes_UB.pdf 
   * L3_intro_to_bayes_UB.pdf 
- Extras:   
   * matplotlibrc (for plotting format purposes)

**INSTALLATION**

No installation is needed. 

Run any code with the -h or --help option to get help on the available options. For example::

    ./moneda_priors.py -h

**QUICK DESCRIPTION**

* L1_progs - Lesson 1 programs

  * moneda_priors.py - Shows the posterior for the coin example, computed with upto three pre-defined priors, with increasing sample size N, for a given coin bias.

  * moneda_ex.py - Shows the posterior for the coin example for a given number of heads and total tosses. As in moneda_priors.py, different priors can be selected and plotted together.

**EXAMPLES**

For the coin example we can see the effect of choosing different priors, using the -p option::

    ./moneda_priors.py -p 1 2 3 -hc 0.6 

This will show the results using pre-defined priors 1, 2 and 3 (uniform, gaussian and spikey).

The posterior median can be plotted with the -m flag and the output save with -f and -o::

    ./moneda_priors.py -p 1 2 3 -hc 0.6  -m -o monedas.png -f

Finally, set a fixed seed for the random number generator with -s::

    ./moneda_priors.py -p 1 2 3 -hc 0.6  -m -o monedas.png -f -s 98765

Attribution
-----------

Cecilia Mateu - cmateu at astrosen.unam.mx

If you have used this code in your research, please let me know and consider acknowledging this package.

License
-------

Copyright (c) 2014 Cecilia Mateu

This is open source and free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see `<http://www.gnu.org/licenses/>`_.
