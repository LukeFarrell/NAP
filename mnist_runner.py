from __future__ import print_function
import numpy as np
from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
import neat
import  matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy import stats
from sklearn.datasets import fetch_mldata
print('FETCHING DATA...')
mnist = fetch_mldata("MNIST original")
print('DATA COMPLETE')
mnist_0 = mnist['data'][0:5000]

def run(G, D, n=None):
    """
    Runs NEAT's genetic algorithm for at most n generations.  If n
    is None, run until solution is found or extinction occurs.
    The user-provided fitness_function must take only two arguments:
        1. The population as a list of (genome id, genome) tuples.
        2. The current configuration object.
    The return value of the fitness function is ignored, but it must assign
    a Python float to the `fitness` member of each genome.
    The fitness function is free to maintain external state, perform
    evaluations in parallel, etc.
    It is assumed that fitness_function does not modify the list of genomes,
    the genomes themselves (apart from updating the fitness member),
    or the configuration object.
    """

    # if G.config.no_fitness_termination and (n is None):
    #     raise RuntimeError("Cannot have no generational limit with no fitness termination")

    # if D.config.no_fitness_termination and (n is None):
    #     raise RuntimeError("Cannot have no generational limit with no fitness termination")

    G_avg_absolute_fitness = []
    D_avg_absolute_fitness = []
    k = 0
    plt.ion()
    while n is None or k < n:
        k += 1

        G.reporters.start_generation(G.generation)
        D.reporters.start_generation(D.generation)

        ################
        # Evaluate all genomes using the user-provided function.
        # fitness_function(list(iteritems(self.population)), self.config)
        size = 784
        iterations = 1
        G_genomes = list(iteritems(G.population))
        D_genomes = list(iteritems(D.population))

        for G_genome_id, G_genome in G_genomes:
            G_genome.fitness = 0.0

        for D_genome_id, D_genome in D_genomes:
            D_genome.fitness = 0.0

        for i in range(iterations):
            #Random noise  for generator
            G_inputs = np.random.uniform(0,255,size)

            for G_genome_id, G_genome in G_genomes:
                G_net = neat.nn.FeedForwardNetwork.create(G_genome, G.config)
                ind_G_outputs = G_net.activate(G_inputs)

                real_input = mnist_0[np.random.choice(len(mnist_0))]

                for D_genome_id, D_genome in D_genomes:
                    D_net = neat.nn.FeedForwardNetwork.create(D_genome, D.config)
                    #Generator Input
                    fake_output = D_net.activate(ind_G_outputs)
                    if fake_output[0] >= 0.5:
                        G_genome.fitness += 1
                    else:
                        D_genome.fitness += 1

                    #Real Input
                    real_output = D_net.activate(real_inputs)
                    #Discriminator detects a real number
                    if real_output[0] >= 0.5:
                        D_genome.fitness += 1

        ####################
        # # Discrimintor output : 1-thinks its real, 0-thinks its fake news
        avg_G_fitness = np.mean([g.fitness for g in itervalues(G.population)])
        avg_D_fitness = np.mean([d.fitness for d in itervalues(D.population)])

        ######################

        # Gather and report statistics.
        D_fitness_list = [d.fitness for d in itervalues(D.population)]
        min_D_fitness = min(D_fitness_list)
        avg_D_fitness = np.mean(D_fitness_list) + min_D_fitness
        # for d in itervalues(D.population):
        #     d.fitness += min_D_fitness

        bestG = None
        for g in itervalues(G.population):
            if bestG is None or g.fitness > bestG.fitness:
                bestG = g
        print("*************GENERATOR**************")
        G.reporters.post_evaluate(G.config, G.population, G.species, bestG)

        bestD = None
        for d in itervalues(D.population):
            if bestD is None or d.fitness > bestD.fitness:
                bestD = d
        print("*************DETECTOR**************")
        D.reporters.post_evaluate(D.config, D.population, D.species, bestD)

        G_net_best = neat.nn.FeedForwardNetwork.create(bestG, G.config)

        D_net = neat.nn.FeedForwardNetwork.create(bestD, D.config)

        G_inputs = np.random.uniform(0,255,size)
        output = G_net_best.activate(G_inputs).reshape(28,28)
        plt.gray()
        plt.imshow(output)
        plt.show()

        # real_1 = np.random.normal(real_mu,real_sigma,num_samples/2)
        # real_2 = np.random.normal(real_mu_2,real_sigma_2,num_samples/2)
        # real = np.concatenate((real_1, real_2))

        # fake_in = np.arange(0,1,(1.0/num_samples)) 
        # fake_out = [G_net.activate([i])[0] for i in fake_in]

        # fake_out_i = [G_net.activate([i])[0] for i in np.arange(0,1,(1.0/num_samples))]
        # absolute_fitness= stats.ks_2samp(real, fake_out_i)[1]
        # G_avg_absolute_fitness.append(np.mean(absolute_fitness))

        # fake_out_i = [D_net.activate([i]*10)[0] for i in np.arange(0,1,(1.0/num_samples))]
        # absolute_fitness= stats.ks_2samp(real, fake_out_i)[1]
        # D_avg_absolute_fitness.append(absolute_fitness)

        # decision_vals = []
        # for i in np.arange(0,1,(1.0/num_samples)):
        #     decision_vals.append(D_net.activate([i]*10)[0]*10)


        # # avg_decision_vals =  []
        # # for i in np.arange(0,1,(1.0/num_samples)):
        # #     intermediate = []
        # #     for d in itervalues(D.population):
        # #         D_net = neat.nn.FeedForwardNetwork.create(d, D.config)
        # #         intermediate.append(D_net.activate([i]*10)[0]*10)
        # #     avg_decision_vals.append(np.mean(intermediate))

        # # weighted_decision_vals =  []
        # # for i in np.arange(0,1,(1.0/num_samples)):
        # #     intermediate = []
        # #     for d in itervalues(D.population):
        # #         D_net = neat.nn.FeedForwardNetwork.create(d, D.config)
        # #         output = D_net.activate([i]*10)[0]*10
        # #         for x in range(int(math.ceil(float(d.fitness)/avg_D_fitness))):
        # #             intermediate.append(output)
        # #     weighted_decision_vals.append(np.mean(intermediate))

        # avg_fake =  []
        # for i in np.arange(0,1,(1.0/num_samples)):
        #     intermediate = []
        #     for g in itervalues(G.population):
        #         G_net = neat.nn.FeedForwardNetwork.create(g, G.config)
        #         intermediate.append(G_net.activate([i])[0])
        #     avg_fake.append(np.mean(intermediate))



        # # print('****************************BEST RATIO', float(bestD.fitness)/avg_D_fitness)
        # if k%1==0:
        #     bins = np.linspace(0,1,100)
        #     try:
        #         fig = sns.distplot(real, color='green', hist=False, bins = bins, kde=True, norm_hist=True)
        #         sns.distplot(fake_out, color='blue', hist=False, bins = bins, kde=True, norm_hist=True)
        #         sns.distplot(avg_fake, color='cyan', hist=False, bins = bins, kde=True, norm_hist=True)

        #         ax = fig.axes
        #         ax.set_ylim(0,20)
        #         ax.set_xlim(0,1)

        #         plt.plot(fake_in, decision_vals, color = 'red')
        #     #     # plt.plot(np.arange(0,1,(1.0/num_samples)), avg_decision_vals, color = 'brown')
        #     #     # plt.plot(np.arange(0,1,(1.0/num_samples)), weighted_decision_vals, color = 'orange')
        #         plt.pause(0.0001)
        #         plt.savefig('./plots/snapshot'+str(k)+'.png')
        #         plt.clf()
        #     except:
        #         print('ERROR')





        # Track the best genome ever seen.
        if G.best_genome is None or bestG.fitness > G.best_genome.fitness:
            G.best_genome = bestG
        if D.best_genome is None or bestD.fitness > D.best_genome.fitness:
            D.best_genome = bestD

        # if not G.config.no_fitness_termination:
        #     # End if the fitness threshold is reached.
        #     fv = G.fitness_criterion(g.fitness for g in itervalues(G.population))
        #     if fv >= G.config.fitness_threshold:
        #         G.reporters.found_solution(G.config, G.generation, bestG)
        #         break

        # if not D.config.no_fitness_termination:
        #     # End if the fitness threshold is reached.
        #     fv = D.fitness_criterion(g.fitness for g in itervalues(D.population))
        #     if fv >= D.config.fitness_threshold:
        #         D.reporters.found_solution(D.config, D.generation, bestD)
        #         break

        # Create the next generation from the current generation.
        G.population = G.reproduction.reproduce(G.config, G.species,
                                                      G.config.pop_size, G.generation)
        D.population = D.reproduction.reproduce(D.config, D.species,
                                                      D.config.pop_size, D.generation)

        # Check for complete extinction.
        if not G.species.species:
            G.reporters.complete_extinction()
            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if G.config.reset_on_extinction:
                G.population = G.reproduction.create_new(G.config.genome_type,
                                                               G.config.genome_config,
                                                               G.config.pop_size)
            else:
                raise CompleteExtinctionException()

        if not D.species.species:
            D.reporters.complete_extinction()
            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if D.config.reset_on_extinction:
                D.population = D.reproduction.create_new(D.config.genome_type,
                                                               D.config.genome_config,
                                                               D.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Divide the new population into species.
        print("*************GENERATOR**************")
        G.species.speciate(G.config, G.population, G.generation)
        G.reporters.end_generation(G.config, G.population, G.species)
        G.generation += 1

        print("*************DECTOR**************")
        D.species.speciate(D.config, D.population, D.generation)
        D.reporters.end_generation(D.config, D.population, D.species)
        D.generation += 1

    # plt.clf()
    # plt.plot(G_avg_absolute_fitness, 'b')
    # plt.savefig('./plots/G_absolute_fitness.png')
    # plt.clf()
    # plt.plot(D_avg_absolute_fitness, 'r')
    # plt.savefig('./plots/D_absolute_fitness.png')
    # plt.clf()

    # if G.config.no_fitness_termination:
    G.reporters.found_solution(G.config, G.generation, G.best_genome)
    # if D.config.no_fitness_termination:
    D.reporters.found_solution(D.config, D.generation, D.best_genome)

    return G.best_genome, D.best_genome