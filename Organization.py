# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:46:09 2020

@author: John Meluso
"""

import pickle
import numpy as np
from numpy.random import default_rng

# Create the random number generator
rng = default_rng()


class Organization(object):
    '''Defines a class Organization which contains an organization network
    structure (a.k.a. an organizational form) populated with agents.'''

    def __init__(self, struct="tree"):
        """
        Creates an instance of class Organization with a specified structure and
        corresponding parameters for that structure. The default is a standard
        tree organizational form.

        Parameters
        ----------
        struct : STRING, optional
            Defines the form or structure of the organization. The
            default is "tree".

        Returns
        -------
        None.

        """

        # Create network graph of organization
        if struct == "tree":

            # Load organization, parents, and siblings from file
            self.org = pickle.load(open("cliquetree_org.pickle","rb"))
            self.Apars = pickle.load(open("cliquetree_parents.pickle","rb"))
            self.Asibs = pickle.load(open("cliquetree_siblings.pickle","rb"))

            # Define other relationships
            self.Agpars = np.matmul(self.Apars,self.Apars)
            self.Akids = np.transpose(self.Apars)
            self.Agkids = np.matmul(self.Akids,self.Akids)

            # Correct grandparent relationship for those without grandparents
            self.Agpars[0:6,0] = np.ones((6))


        else:
            print("Input 'struct' for 'Organization' is not valid.")

        # Initialize list of populations
        self.pops = []

        # Initialize constants
        self.n_nodes = len(self.org.nodes())  # Number of nodes in network
        self.n_pops = 0  # Number of populations added to the org for any time
        self.n_cultatt = 3  # The number of cultural attributes
        self.d_sim = 0  # Dimension for similarity in culture matrix
        self.d_perf = 1  # Dimension for performance in culture matrix
        self.d_inc = 2  # Dimension for inclusiveness in culture matrix
        self.curr_step = 0  # The current timestep in the simulation
        self.pop_membership = []

        # Initialize structures for populating
        self.culture = np.empty([self.n_nodes,self.n_cultatt])

        # Create organization history dictionary
        self.history = {'aff_sim': [],
                        'aff_perf': [],
                        'aff_inc': [],
                        'soc_sim': [],
                        'soc_perf': [],
                        'soc_inc': [],
                        'perf_mean': []}


    def __repr__(self):
        '''Returns a representation of the organization'''
        return self.__class__.__name__


    def add_pop(self,starting=1,hires=1,aff_dist="beta",aff_sim=0.25,
                aff_perf=0.25,aff_inc=0.5,aff_var=15,perf_dist="beta",
                perf_mean=0.5,perf_var=15):
        """
        Adds a population to the organization instance.

        Parameters
        ----------
        starting : [0,1], optional
            Specifies the probability that a member of the starting organization
            will be from this population. All probabilities must sum to 1. The
            default is 1.
        hires : [0,1], optional
            Specifies the probability that a new hire will will be from this
            population. All probabilities must sum to 1. The default is 1.
        aff_dist : STRING, optional
            The culture distribution type of the population, either "beta" or
            "uniform". The default is "beta".
        aff_sim : [0.1,0.9], optional
            The mean of the sampling distribution for an agent's affinity for
            cultural similarity. Applies to only beta distributions. The default
            is 0.25.
        aff_perf : [0.1,0.9], optional
            The mean of the sampling distribution for an agent's affinity for
            performance. Applies to only beta distributions. The default is
            0.25.
        aff_inc : [0.1,0.9], optional
            The mean of the sampling distribution for an agent's affinity for
            inclusiveness. Applies to only beta distributions. The default is
            0.25.
        aff_var : [0.1,0.9], optional
            The variance of the culture beta distribution. Applies only to beta
            distributions. The default is 15.
        perf_dist : STRING, optional
            The performance distribution type of the population, either "beta"
            or "uniform". The default is "beta".
        perf_mean : [0.1,0.9], optional
            The mean of the sampling distribution for an agent's performance.
            Applies only to beta distributions. The default is 0.5.
        perf_var : (0,inf), optional
            The variance of the performance beta distribution. Applies only to
            beta distributions. The default is 15.

        Returns
        -------
        None.

        """

        new_pop = {'index': len(self.pops),
                   'rep_start': starting,
                   'rep_gen': hires,
                   'aff_dist': aff_dist,
                   'aff_sim': aff_sim,
                   'aff_perf': aff_perf,
                   'aff_inc': aff_inc,
                   'aff_var': aff_var,
                   'perf_dist': perf_dist,
                   'perf_mean': perf_mean,
                   'perf_var': perf_var
                   }
        self.pops.append(new_pop)
        self.n_pops += 1


    def fill_org(self):
        """Fills the organization with members of the starting population.
        Assumes that the organization has already been initialized and either
        one or two populations have been added."""

        # Determine starting population from existing populations
        self.pop_membership = rng.choice(a=np.arange(self.n_pops),
            size=self.n_nodes,
            p=[self.pops[x_]['rep_start'] for x_ in np.arange(self.n_pops)])

        # Populate the culture values of each node






        # # Define culture
        # self.culture[:,self.d_sim] = rng.beta( \
        #           beta_a(),
        #           beta_b())







    def run_org(self,steps=200):
        """Steps the organization forward in time for a specified number of
        turns. Assumes that the organization has already been filled."""





    def record_data(self):
        """Records variables to the history dictionary."""





def beta_a(mu,phi):
    """Transforms beta function parameters from average and variance form to
    the alpha parameter"""
    a = mu*phi
    return a


def beta_b(mu,phi):
    """Transforms beta function parameters from average and variance form to
    the beta parameter"""
    b = (1-mu)*phi
    return b


if __name__ == '__main__':

    org_test = Organization()

    # Add generic org of beta culture distribution
    org_test.add_pop(starting=1,
                 hires=0.8,
                 aff_dist="uniform",
                 aff_sim=0.25,
                 aff_perf=0.25,
                 aff_inc=0.5)

    # Add generic org of uniform culture distribution
    org_test.add_pop(starting=0,
                 hires=0.2,
                 aff_dist="uniform")

    # Populate the organization from the specified populations
    org_test.fill_org()