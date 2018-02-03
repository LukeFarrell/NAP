from __future__ import print_function
import os
import neat
import visualize
import numpy as np
import mnist_runner as runner
import time 
os.environ["PATH"] += "C:/Users/Luke Farrell/Anaconda2/Library/bin/graphviz/"

def run2(G_config, D_config):
    # Load configuration.
    G_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         G_config)
    D_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     D_config)

    # Create the population, which is the top-level object for a NEAT run.
    G = neat.Population(G_config)
    D = neat.Population(D_config)

    # Add a stdout reporter to show progress in the terminal.
    G.add_reporter(neat.StdOutReporter(True))
    G_stats = neat.StatisticsReporter()
    G.add_reporter(G_stats)
    G.add_reporter(neat.Checkpointer(5))

    D.add_reporter(neat.StdOutReporter(True))
    D_stats = neat.StatisticsReporter()
    D.add_reporter(D_stats)
    D.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    G_winner, D_winner = runner.run(G, D, 1000)

    # Display the winning genome.
    print('\nBest Generator:\n{!s}'.format(G_winner))
    print('\nBest Detector:\n{!s}'.format(D_winner))

    # Show output of the most fit genome against training data.
    print('\nOutput G:')
    G_winner_net = neat.nn.FeedForwardNetwork.create(G_winner, G_config)
    D_winner_net = neat.nn.FeedForwardNetwork.create(D_winner, D_config)


    # for xi, xo in zip(xor_inputs, xor_outputs):
    #     output = winner_net.activate(xi)
    #     print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {0:'OUTPUT'}
    os.environ["PATH"] = "C:/Users/Luke Farrell/Anaconda2/Library/bin/graphviz/"
    visualize.draw_net(G_config, G_winner, True, node_names=node_names, filename = './plots/G_net.png')
    visualize.plot_stats(G_stats, ylog=False, view=True, filename = './plots/G_stats.png')
    visualize.plot_species(G_stats, view=True, filename='./plots/G_evo.png')

    node_names = {0:'OUTPUT'}
    visualize.draw_net(D_config, D_winner, True, node_names=node_names, filename = './plots/D_net.png')
    visualize.plot_stats(D_stats, ylog=False, view=True, filename = './plots/D_stats.png')
    visualize.plot_species(D_stats, view=True, filename='./plots/D_evo.png')

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # G.run(eval_genomes, 10)


###################################



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    G_config_path = os.path.join(local_dir, 'G_Config_MNIST')
    D_config_path = os.path.join(local_dir, 'D_Config_MNIST')
    run2(G_config_path, D_config_path)