# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:07:49 2020

@author: John Meluso
"""

import sys, os
import numpy as np
import datetime as dt
import Organization as og
import data_manager as dm


class Simulation(object):
    """Class Simulation which includes one run of each parameter set."""

    def __init__(self,n_steps):
        """Creates an instance of class Simulation."""

        # Create simulation spec parameters
        self.index_pops = 0
        self.index_mode = 1
        self.index_p1_culture = 2
        self.index_p1_start = 3
        self.index_p1_hire = 4
        self.index_p2_culture = 5

        # Create cases
        self.cases = []
        self.create_sim_params()
        self.n_steps = n_steps

    def create_sim_params(self):
        """Create one instance of each combination of the MCC simulation run
        parameters. Case Structure appears as follows:

            [n_pops,pop_mode,pop1_culture,pop2_culture,pop_start,pop_hire]

        """

        """Create 1 population uniform cases"""
        n_pops = 1
        pop_mode = "uniform_2var"
        new_case = [n_pops,pop_mode]
        self.cases.append(new_case)

        """Create 1 population beta cases"""
        n_pops = 1
        pop_mode = "beta_2var"
        pop1_culture = np.linspace(0.1,1.0,9,endpoint=False)
        for cc in pop1_culture:
            new_case = [n_pops,pop_mode,cc]
            self.cases.append(new_case)

        """Create 2 population beta cases"""
        n_pops = 2
        pop_mode = "beta_2var"
        pop_start = np.linspace(0.5,1.0,5,endpoint=False)
        pop1_hire_limit = 1.0  # for arange in loop
        pop1_culture = np.linspace(0.5,1.0,5,endpoint=False)
        pop2_culture = np.linspace(0.1,1.0,9,endpoint=False)

        # Loop through all starting fractions and hiring fractions
        for ss in pop_start:
            pop2_steps = int(np.round((pop1_hire_limit - ss)/0.1))
            for hh in np.linspace(ss,pop1_hire_limit,pop2_steps,
                                  endpoint=False):

                # Loop through full rectangle of population 1 and 2 cultures
                for cc in pop1_culture:
                    for dd in pop2_culture:
                        if not(abs(cc - dd) < 1E-4):

                            # Construct cases
                            new_case = [n_pops,pop_mode,cc,dd,ss,hh]
                            self.cases.append(new_case)

                # Loop through special cases for populations 1 and 2 cultures
                # w/ 0.3,0.5 and 0.3,0.7 for pop1_culture,pop2_culture resp.
                new_case = [n_pops,pop_mode,0.3,0.5,ss,hh]
                self.cases.append(new_case)
                new_case = [n_pops,pop_mode,0.3,0.7,ss,hh]
                self.cases.append(new_case)

    def run_Organization(self,tt):
        """Creates and runs an organization for a specified number of populations
         with a set of input parameters for each population and for a specified
        number of steps."""

        # Create organization and empty population array
        org = og.Organization()

        # Add the test populations
        pops = self.add_2var_pops(tt)

        # Populate the organization with the culture, performance, and populations
        org.fill_org(pops)

        # Run organization for n steps
        org.org_step(self.n_steps)

        # Return results
        return org.return_results()


    def add_2var_pops(self,tt):
        """Adds specified number of populations (1 or 2 only) with a set of
        parameters for each population."""


        # Copy mode parameters to variables for input into populations
        n_pops = self.cases[tt][self.index_pops]
        pop_mode = self.cases[tt][self.index_mode]

        pops = []  # Create empty population array

        # Assign rest of variables based on pop_mode
        if pop_mode == "uniform_2var":
            pops.append(og.Population(aff_dist=pop_mode))

        elif (n_pops == 1) and (pop_mode == "beta_2var"):
            pop1_culture = self.cases[tt][self.index_p1_culture]
            pops.append(og.Population(aff_dist=pop_mode,
                                      aff_sim=pop1_culture/2,
                                      aff_perf=pop1_culture/2,
                                      aff_inc=1-pop1_culture))

        elif n_pops == 2:
            pop1_culture = self.cases[tt][self.index_p1_culture]
            pop1_start = self.cases[tt][self.index_p1_start]
            pop1_hire = self.cases[tt][self.index_p1_hire]
            pop2_culture = self.cases[tt][self.index_p2_culture]
            pops.append(og.Population(starting=pop1_start,
                                      hires=pop1_hire,
                                      aff_dist=pop_mode,
                                      aff_sim=pop1_culture/2,
                                      aff_perf=pop1_culture/2,
                                      aff_inc=1-pop1_culture))
            pops.append(og.Population(starting=1-pop1_start,
                                      hires=1-pop1_hire,
                                      aff_dist=pop_mode,
                                      aff_sim=pop2_culture/2,
                                      aff_perf=pop2_culture/2,
                                      aff_inc=1-pop2_culture))
        else:
            print("Case " + str(tt) + " is not a valid population.")

        return pops


if __name__ == '__main__':

    # Get start time
    t_start = dt.datetime.now()

    if sys.platform.startswith('linux'):

        # get the number of this job and the total number of jobs from the PBS
        # queue system. These arguments are given by the VACC to this script via
        # submit_job.sh. If there are n jobs to be run total (numruns = n), then
        # runnum should run from 0 to n-1. In notation: [0,n) or [0,n-1].
        try:
            runnum, numruns = map(int, sys.argv[1:])
        except IndexError:
            sys.exit("Usage: %s runnum numruns" % sys.argv[0] )
        output_dir = '/gpfs2/scratch/jmeluso/culture_sim/data/'

    else:

        runnum = 0
        numruns = 1
        output_dir = 'data/'

    # Create data storage directory if it doesn't already exist


    # Main execution code for one instance of the MCC simulation
    n_steps = 100
    sim = Simulation(n_steps)
    for case in np.arange(len(sim.cases)):

        # Run simulation for specified set of parameters
        results = sim.run_Organization(case)

        # Build name for specific test
        case = f'case{case:04}'
        job = f'run{runnum:04}'
        time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        fileext = '.npy'
        filename = output_dir + case + '_' + job + fileext

        # Save results to location specified by platform
        dm.save_mcc(results,filename)

    # Print end time
    t_stop = dt.datetime.now()
    print(t_stop - t_start)
