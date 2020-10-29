# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:46:09 2020

@author: John Meluso
"""

import pickle
import numpy as np
import numpy.linalg as la
from numpy.random import default_rng

# Create the random number generator
rng = default_rng()


class Organization(object):
    """Defines a class Organization which contains an organization network
    structure (a.k.a. an organizational form) populated with agents."""

    def __init__(self, struct="tree"):
        """Creates an instance of class Organization with a specified structure
        and corresponding parameters for that structure. The default is a
        standard tree organizational form.

        Parameters
        ----------
        struct : STRING, optional
            Defines the form or structure of the organization. The
            default is "tree".
        pops : Population, required
            One or more populations provided to the organization in an array of
            populations.

        Returns
        -------
        None.

        """

        # Set org structure
        self.struct = struct

        # Create network graph of organization
        if self.struct == "tree":

            # Load organization, parents, and siblings from file
            self.org = pickle.load(open("cliquetree_org.pickle","rb"))
            self.A_pars = pickle.load(open("cliquetree_parents.pickle","rb"))
            self.A_sibs = pickle.load(open("cliquetree_siblings.pickle","rb"))

            # Define other relationships
            self.A_gpars = np.matmul(self.A_pars,self.A_pars)
            self.A_kids = np.transpose(self.A_pars)
            self.A_gkids = np.matmul(self.A_kids,self.A_kids)

            # Correct grandparent relationship for those without grandparents
            self.A_gpars[0:6,0] = np.ones((6))

        else:
            print("Input 'struct' for 'Organization' is not valid.")

        """Population Variables"""
        self.pops = []  # List of populations for the org
        self.from_pop = []  # Array of populations that current employees are from

        """Network Count Parameters"""
        # For nodes, parents, grandparents, siblings,kids, and grandkids. No
        # values are allowed to be zero because they're mostly used as
        # divisors and the matrices will be zero in those cases.
        self.n_nodes = len(self.org.nodes())
        self.id = np.identity(self.n_nodes)
        self.norm_pars = np.divide(self.id,np.sum(self.A_pars,axis=1) \
            + np.array(np.sum(self.A_pars,axis=1) == 0))
        self.norm_gpars = np.divide(self.id,np.sum(self.A_gpars,axis=1) \
            + np.array(np.sum(self.A_gpars,axis=1) == 0))
        self.norm_sibs = np.divide(self.id,np.sum(self.A_sibs,axis=1) \
            + np.array(np.sum(self.A_sibs,axis=1) == 0))
        self.norm_kids = np.divide(self.id,np.sum(self.A_kids,axis=1) \
            + np.array(np.sum(self.A_kids,axis=1) == 0))
        self.norm_gkids = np.divide(self.id,np.sum(self.A_gkids,axis=1) \
            + np.array(np.sum(self.A_gkids,axis=1) == 0))

        """Unit Vectors"""
        self.unit_x = np.array([1,0,0])
        self.unit_y = np.array([0,1,0])
        self.unit_z = np.array([0,0,1])

        """Normalizing Parameters"""
        # Normalization divisors for socialization, branch, and promotion calcs
        self.norm_soc = np.divide(self.id,np.ones([self.n_nodes,1]) \
            + np.array(np.sum(self.A_pars,axis=1) > 0) \
            + np.array(np.sum(self.A_sibs,axis=1) > 0))
        self.norm_branch = np.divide(self.id,np.ones([self.n_nodes,1]) \
            + np.array(np.sum(self.A_kids,axis=1) > 0))
        self.norm_prom = np.divide(self.id,np.ones([self.n_nodes,1]) \
            + np.array(np.sum(self.A_gpars,axis=1) > 0))

        """Culture Parameters & Variables"""
        self.n_cultatt = 3  # Number of culture attributes
        self.index_sim = 0  # Similarity index
        self.index_perf = 1  # Performance index
        self.index_inc = 2  # Inclusiveness index
        self.culture = np.empty([self.n_nodes,self.n_cultatt])

        """Performance Parameters & Variables"""
        self.n_perfatt = 2  # Number of performance attributes for beta fns
        self.index_mean = 0  # Performance mean index
        self.index_disp = 1  # Performance dispersion (like variance) index
        self.perf_params = np.zeros([self.n_nodes,self.n_perfatt])
        self.perf_indiv = np.zeros([self.n_nodes,])
        self.perf_branch = np.zeros([self.n_nodes,])

        """Promotion Parameters & Variables"""
        self.prom_fit = np.zeros([self.n_nodes,self.n_cultatt])
        self.prom_score = np.zeros([self.n_nodes,])

        """Retirement Parameters & Variables"""
        self.n_retire_opts = 2
        self.retire_prob = 0.2
        self.empty_positions = np.zeros([self.n_nodes,])

    def __repr__(self):
        """Returns a representation of the organization"""
        return self.__class__.__name__

    def fill_org(self, pops):
        """
        Populates the culture and performance parameters for each member of the
        organization given a set of populations.

        Parameters
        ----------
        pops : Population Array
            An array of one or more populations for use in organization
            initialization and hiring.

        Returns
        -------
        None.

        """

        # Add populations to the organization for hiring
        self.pops = pops
        self.add_employee()

        # Initialize structures for populating culture
        self.add_culture()
        self.social = self.culture

        # Initialize sutructures for populating performance
        self.add_performance()

    def add_employee(self, loc=-1):
        """Adds one or more employees to the organization by sampling from
        population probabilities. Either creates one employee at a location
        (loc) or all employees (-1)."""
        if loc > -1:
            self.from_pop[loc] = rng.choice(a=len(self.pops),
                p=[self.pops[ii].rep_gen for ii in np.arange(len(self.pops))])
        else: # assume all nodes
            self.from_pop = rng.choice(a=len(self.pops),size=self.n_nodes,
                 p=[self.pops[ii].rep_start for ii \
                    in np.arange(len(self.pops))])

    def add_culture(self,loc=-1):
        """Creates culture matrix for all the nodes from populations (-1), or
        adds culture for a single specified node (loc).

        CULTURE DETERMINATION RULES:
        x = similarity [0,1]
        y = performance [0,1]
        z = inclusiveness [0,1]

        FOR MCC SIM:
        x + y + z = 1
        x = y
        x = (1 - z)/2

        Therefore, sample z, calculate x & y from z
        """

        # Generate range of nodes to update
        if loc > -1:  # Just get the one node
            node_range = np.arange(loc,loc+1)
        else:  # Get all nodes
            node_range = np.arange(self.n_nodes)

        # Generate culture values by first cycling through the nodes
        for ii in node_range:

            # CASE 1: linear_2var
            if self.pops[self.from_pop[ii]].aff_dist == "linear_2var":

                # Sample z from a LINEAR UNIFORM distribution
                self.culture[ii,:] = np.array([linear_uniform()])

            # CASE 2: beta_2var
            elif self.pops[self.from_pop[ii]].aff_dist == "beta_2var":

                # Sample z form a LINEAR BETA distribution with mean at
                # aff_inc and var at aff_var.
                self.culture[ii,:] = np.array([linear_beta(
                    self.pops[self.from_pop[ii]].aff_inc,
                    self.pops[self.from_pop[ii]].aff_var,
                    )])

            # CASE 3: beta_3var
            elif self.pops[self.from_pop[ii]].aff_dist == "beta_3var":

                # Sample x from a TRIANGULAR BETA distribution with means at
                # aff_sim & aff_perf, and both vars at aff_var.
                self.culture[ii,:] = np.array([triangle_beta(
                    self.pops[self.from_pop[ii]].aff_sim,
                    self.pops[self.from_pop[ii]].aff_var,
                    self.pops[self.from_pop[ii]].aff_perf,
                    self.pops[self.from_pop[ii]].aff_var,
                    )])

            # CASE 4: "linear_3var"
            else:

                # Sample z from a TRIANGULAR UNIFORM distribution
                self.culture[ii,:] = np.array([triangle_uniform()])

    def add_performance(self,loc=-1):
        """Adds performance matrix for either one (loc) or all (-1) nodes
        from the populations."""

        # Generate range of nodes to update
        if loc > -1:  # Just get the one node
            node_range = np.arange(loc,loc+1)
        else:  # Get all nodes
            node_range = np.arange(self.n_nodes)

        # Generate performance values by cycling through the nodes
        for ii in node_range:

            # Draw a performance distribution mean for each employee
            beta_a, beta_b = beta(self.pops[self.from_pop[ii]].perf_mean,
                              self.pops[self.from_pop[ii]].perf_var)
            self.perf_params[ii,self.index_mean] = rng.beta(beta_a, beta_b)

            # Set performance dispersion for each employee
            self.perf_params[ii,self.index_disp] = \
                self.pops[self.from_pop[ii]].perf_var

    def org_step(self,n_steps = 1):
        """Steps the organization forward in time a specified number of steps,
        and otherwise defaults to one step. Assumes that the organization has
        already been filled."""

        # Create history structure for the number of nodes and steps
        self.history = History(n_steps,self.n_nodes,self.n_cultatt,
                               self.n_perfatt)

        for st in np.arange(n_steps):

            # Socialize agents
            self.socialize()

            # Update individual performances
            self.perform_individuals()

            # Calculate branch performances by reverse iteration
            self.perform_branches()

            # Calculate promotion fitnesses & scores
            self.calc_promotion_fitness()
            self.calc_promotion_scores()

            # Record History from turn (promotion/hiring reflect in next step)
            self.history.record_history(st, self.from_pop, self.culture,
                self.social, self.perf_params, self.perf_indiv,
                self.perf_branch, self.prom_fit, self.prom_score)

            # Perform retirement
            self.gen_retire()

            # Perform promotion & hiring
            self.emp_fill()

    def socialize(self):
        """Socialization function."""
        term_pars = np.matmul(self.norm_pars,
                               np.matmul(self.A_pars,self.social))
        term_sibs = np.matmul(self.norm_sibs,
                               np.matmul(self.A_sibs,self.social))
        self.social = np.matmul(self.norm_soc,
                                 self.culture + term_pars + term_sibs)

    def perform_individuals(self):
        """Generate performance of individuals"""

        # Generate performance values by first cycling through the nodes
        for ii in np.arange(self.n_nodes):

            # Next, check its distribution type
            if self.pops[self.from_pop[ii]].aff_dist == "uniform":

                # Sample perf_indiv from a UNIFORM distribution
                self.perf_indiv[ii] = rng.uniform()

            else:  # Otherwise defaults to beta distribution

                # Else sample perf_indiv from a BETA distribution
                beta_a, beta_b = beta(self.perf_params[ii,self.index_mean],
                          self.perf_params[ii,self.index_disp])
                self.perf_indiv[ii] = rng.beta(beta_a, beta_b)

    def perform_branches(self):
        """Generate performance for branches in reverse. NOTE: Currently
        calculated in reverse from last created node to first node to ensure
        that parent nodes include branch performances of children."""

        # Calculate branch performance values by first cycling through nodes
        for ii in np.arange(self.n_nodes-1,-1,-1):

            # Calculate branch performance for each node
            term_kids = self.norm_kids[ii,ii] \
                * np.matmul(self.A_kids[ii,:],self.perf_branch)
            self.perf_branch[ii] = self.norm_branch[ii,ii] \
                * (self.perf_indiv[ii] + term_kids)

    def calc_promotion_fitness(self):
        """Calculates the promotion fitness for each node. NOTE: Currently
        calculates similarity term as an average of the culture of all parents,
        which may not be appropriate for all promotion algorithms."""

        # Calculate vectors for populating promotion fitness matrix
        term_sim = np.ones(self.n_nodes) \
            - la.norm(x = np.matmul(self.norm_gpars,
                                    np.matmul(self.A_gpars,self.social)) \
                      - self.social,axis = 1)
        term_perf = self.perf_branch
        term_inc = np.matmul(self.culture,self.unit_z)

        # Compile promotion fitness matrix
        self.prom_fit = np.stack((term_sim,term_perf,term_inc),axis=-1)

    def calc_promotion_scores(self):
        """Calculates the promotion score for each node. Make sure to use the
        copy() method if using np.diag or np.diagonal, which returns a read/
        write view starting with NumPy 1.10."""
        self.prom_score = np.diag(np.matmul(np.matmul(
            self.A_gpars,self.social),np.transpose(self.prom_fit))).copy()

    def gen_retire(self):
        """Generates the members of the population to retire with a specified
        probability."""
        self.empty_positions = rng.choice(a=self.n_retire_opts,
            size=self.n_nodes, p=[1-self.retire_prob,self.retire_prob])

    def emp_fill(self):
        """Promote non-retiring member into openings from grandchildren."""

        # Loop through nodes from top down to find the ones that are empty
        for ii in np.arange(self.n_nodes):

            # Only perform actions for empty positions
            if self.empty_positions[ii] == 1:

                # Reset potentially promotable options
                filled = False
                A_prom = self.A_kids

                # Loop through until the empty position has been filled
                while not(filled):

                    # If employees exist in the selected generation
                    if np.sum(A_prom[ii,:])>0:

                        # If at least one employee is promotable
                        if np.dot(A_prom[ii,:],1-self.empty_positions)>0:

                            # Get the location of the most qualified employee
                            emp_to_prom = np.argmax(A_prom[ii,:] \
                                 * self.prom_score * (1 - self.empty_positions))

                            # Promote that employee
                            self.emp_prom(emp_to_prom,ii)
                            filled = True

                        # Otherwise, no employees in generation are promotable
                        else:

                            # So go to the next generation (get children)
                            A_prom = self.A_kids @ A_prom

                    # No employees exist in generation (no children)
                    else:

                        # So hire a new employee to the position
                        self.emp_hire(ii)
                        filled = True

    def emp_prom(self,loc_from,loc_to):
        """Promote an employee from one location to another."""

        # Populate new location
        self.culture[loc_to,:] = self.culture[loc_from,:]
        self.social[loc_to,:] = self.social[loc_from,:]
        self.from_pop[loc_to] = self.from_pop[loc_from]
        self.perf_params[loc_to,:] = self.perf_params[loc_from,:]

        # Clear original location
        self.culture[loc_from,:] = np.zeros(self.n_cultatt)
        self.from_pop[loc_from] = -1
        self.perf_branch[loc_from] = 0
        self.perf_indiv[loc_from] = 0
        self.perf_params[loc_from,:] = np.zeros(self.n_perfatt)
        self.prom_fit[loc_from,:] = np.zeros(self.n_cultatt)
        self.prom_score[loc_from] = 0
        self.social[loc_from,:] = np.zeros(self.n_cultatt)

        # Set location as needing to be filled
        self.empty_positions[loc_from] = 1

    def emp_hire(self,loc_to):
        """Hire new employees into opening by population sampling."""

        # Pick a new employee from possible populations
        self.add_employee(loc_to)

        # Generate initial culture for that employee
        self.add_culture(loc_to)
        self.social[loc_to,:] = self.culture[loc_to,:]

        # Generate performance parameters for that employee
        self.add_performance(loc_to)

        # Set all performance values to zero for now
        self.perf_branch[loc_to] = 0
        self.perf_indiv[loc_to] = 0
        self.prom_fit[loc_to,:] = np.zeros(self.n_cultatt)
        self.prom_score[loc_to] = 0

    def return_results(self):
        """Return the history of the organization."""
        return self.history


class Population(object):
    """Defines an instances of class population from which the organization can
    sample either to start or as new hires."""


    def __init__(self,starting=1,hires=1,aff_dist="beta_2var",aff_sim=0.25,
                aff_perf=0.25,aff_inc=0.5,aff_var=15,perf_dist="beta",
                perf_mean=0.5,perf_var=15):
        """
        Initializes an instance of class population.

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

        self.rep_start = starting
        self.rep_gen = hires
        self.aff_dist = aff_dist
        self.aff_sim = aff_sim
        self.aff_perf = aff_perf
        self.aff_inc = aff_inc
        self.aff_var = aff_var
        self.perf_dist = perf_dist
        self.perf_mean = perf_mean
        self.perf_var = perf_var


class History(object):
    """Instance of a history structure for holding results. Contains structures
    for demographics, culture (including similarity, performance, and
    inclusiveness), socialization (including similarity, performance, and
    inclusiveness), and performance (including individual and branch scores)."""

    def __init__(self,n_steps,n_nodes,n_cultatt,n_perfatt):

        # Create organization history arrays and dictionaries
        self.demographics = np.zeros((n_steps,n_nodes))
        self.culture = np.zeros((n_steps,n_nodes,n_cultatt))
        self.socialization = np.zeros((n_steps,n_nodes,n_cultatt))
        self.performance_params = np.zeros((n_steps,n_nodes,n_perfatt))
        self.performance_indiv = np.zeros((n_steps,n_nodes))
        self.performance_branch = np.zeros((n_steps,n_nodes))
        self.promotion_fitness = np.zeros((n_steps,n_nodes,n_cultatt))
        self.promotion_score = np.zeros((n_steps,n_nodes))

    def record_history(self,step,demo,cult,soc,perf_par,perf_ind,perf_bra,
                       prom_fit,prom_sco):
        self.demographics[step,:] = demo.copy()
        self.culture[step,:,:] = cult.copy()
        self.socialization[step,:,:] = soc.copy()
        self.performance_params[step,:,:] = perf_par.copy()
        self.performance_indiv[step,:] = perf_ind.copy()
        self.performance_branch[step,:] = perf_bra.copy()
        self.promotion_fitness[step,:,:] = prom_fit.copy()
        self.promotion_score[step,:] = prom_sco.copy()


def beta(mu,phi):
    """Transforms beta function parameters from average and variance form to
    the alpha & beta parameters"""
    a = mu*phi
    b = (1-mu)*phi
    return a, b


def linear_uniform():
    """Generates one uniformly distributed random value and calculates two
    other equal values, the three of which sum to one (2x + z = 1). First
    transforms the mu and phi into a and b parameters for the beta function."""
    z = rng.uniform()
    x = (1 - z)/2
    y = x

    return x, y, z


def linear_beta(mu,phi):
    """Generates one beta distributed random value and calculates two other
    equal values, the three of which sum to one (2x + z = 1). First transforms
    the mu and phi into a and b parameters for the beta function."""
    a, b = beta(mu,phi)
    z = rng.beta(a, b)
    x = (1 - z)/2
    y = x

    return x, y, z


def triangle_uniform():
    """Generates three uniformly random values that sum to one via triangle
    point picking (see the following website for more details on the math:
    https://mathworld.wolfram.com/TrianglePointPicking.html), Randomly draws
    two values x and y on [0,1] and converts any values of x and y such that
    x + y > 1 into values such that x + y < 1."""
    x = rng.uniform()
    y = rng.uniform()
    if x + y > 1:
        x = 1 - x
        y = 1 - y
    z = 1 - x - y

    return x, y, z


def triangle_beta(mu1,phi1,mu2,phi2):
    """Generates three beta distributed random values that sum to one via
    triangle point picking (see the following website for more details on the
    math: https://mathworld.wolfram.com/TrianglePointPicking.html), Randomly
    draws two values x and y on [0,1] and converts any values of x and y such
    that x + y > 1 into values such that x + y < 1."""
    a1, b1 = beta(mu1,phi1)
    a2, b2 = beta(mu2,phi2)
    valid = False
    while not(valid):
        x = rng.beta(a1,b1)
        y = rng.beta(a2,b2)
        if x + y <= 1:
            valid = True
            z = 1 - x - y

    return x, y, z


if __name__ == '__main__':

    org_test = Organization()
