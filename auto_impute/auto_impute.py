# James Allingham
# Feb 2018
# auto_impute.py
# Main file for AutoImpute CLI

import argparse
import numpy as np
from sys import stdout, stderr

import csv_reader
import mi
import sg
import gmm
import dp
import mixed

from utilities import print_err

def main(args):
    # set random seed
    if args.rand_seed: np.random.seed(args.rand_seed)

    reader = csv_reader.CSVReader(args.file, args.delimiter, args.header, args.indicator)

    data = reader.get_masked_data()

    if args.verbose:
        print_err("Read the file %s with %s rows and %s columns." % ((args.file, ) + data.shape))
        out_str = "Number missing elements: "
        for i in range(data.shape[1]):
            out_str += "col %s: %s  " % (i, np.sum(data.mask[:, i]))
        print_err(out_str)
        print_err("Percentage missing elements: %s\n" % (np.mean(data.mask),))
        print_err("")

    if args.gaussian_mixture:
        model = gmm.GMM(data, args.num_components, verbose=args.verbose, map_est=not args.ml_estimation)
        model.fit(max_iters=args.max_iters, ϵ=args.epsilon)
    elif args.single_gaussian:
        model = sg.SingleGaussian(data, verbose=args.verbose)
        model.fit(max_iters=args.max_iters, ϵ=args.epsilon)
    elif args.dirichlet_process:
        model = dp.DP(data, verbose=args.verbose)
    elif args.mixed_dp_gmm:
        model = mixed.Mixed(data, num_components=args.num_components, verbose=args.verbose, assignments=args.column_assignments)
    else:
        model = mi.MeanImpute(data, verbose=args.verbose)

    print_err("Avg LL: %s" % model.log_likelihood(return_mean=True, complete=False))
    print_err("")
    if args.test is not None:
        test_data = np.genfromtxt(args.test, delimiter=args.delimiter)
        imputed_X =  model.ml_imputation()
        print_err("RMSE: %s" % np.sqrt(np.mean(np.power(test_data - imputed_X, 2))))
        print_err("")

    # set formatting
    # np.set_printoptions(formatter={'float': args.format})
    if args.mixed_dp_gmm and args.verbose:
        continuous_probs = model._continuous_probs()
        np.savetxt(stderr, continuous_probs, fmt=args.format)

    # Write the output to file
    # if there was no name supplied then write to std out
    if args.file_name == None:
        ofile = stdout
        print_err("Repaired file(s) written to std out.")
        print_err("")
    else:
        ofile = args.file_name

    # either sample the results or get the maximim likelihood imputation
    if args.sample:
        result = model.sample(args.sample)
    else:
        result = model.ml_imputation()

    # build a key word arguments dict
    kwargs = {"delimiter": args.delimiter, "fmt":args.format}
    if args.header:
        kwargs["header"] = reader.get_column_names

    if args.sample:
        for s in range(args.sample):
            tmp_ofile = ofile
            if type(ofile) == str:
                tmp_ofile += "." + str(s)
                print_err("Writing repaired file to %s" % tmp_ofile)
            else:
                print("")
            np.savetxt(tmp_ofile, result[s], **kwargs)
    else:
        if type(ofile) == str:
            print_err("Writing repaired file to %s." % ofile)
        np.savetxt(ofile, result, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically repairs CSV files with missing entries")

    parser.add_argument("file", help="name of the file to repair")
    parser.add_argument("-v", "--verbose", help="increase the output verbosity",
                        action="store_true")
    parser.add_argument("-d", "--delimiter", help="file delimiter (default: ',')",
                        type=str, default=",")
    parser.add_argument("-hd", "--header", help="use the first row as column names (default: False)",
                        type=bool, default=False)
    parser.add_argument("-rs", "--rand_seed", help="random seed to use (default: None)",
                        type=int)
    parser.add_argument("-t", "--test", help="file to use for calculating RMSE",
                        type=str, default=None)
    parser.add_argument("-i", "--indicator", help="inidcator string that a value is missing (default: '')",
                        type=str, default='')
    parser.add_argument("-k", "--num_components", help="number of components for mixture models (default: 10)",
                        type=int, default=10)
    parser.add_argument("-a", "--column_assignments", help="data type assignments for each column either 'r' for real or 'd' for discrete e.g. 'dddrrr' for 3 discrete followed by 3 real (default: all real)",
                        type=str, default=None)
    parser.add_argument("-e", "--epsilon", help="ϵ (model stopping criterion): if LL_new - LL_old < ϵ then stop iterating (default: 1e-1)",
                        type=float, default=1e-1)
    parser.add_argument("-n", "--max_iters", help="maximum number of iterations to fit model (default: 100)",
                        type=int, default=100)
    parser.add_argument("-mle", "--ml_estimation", help="use MLE rather than MAP for non-Bayesian models (default: False)",
                        action="store_true")
    parser.add_argument("-o", "--file_name", help="file name to write repaired files to (default: none, print to std out)",
                        type=str, default=None)
    parser.add_argument("-fmt", "--format", help="Format for printing floating point numbers (default: '%.5f')",
                        type=str, default="%.5g")

    model_group = parser.add_mutually_exclusive_group()
    # speed_group.add_argument("-f", "--fast", help="quick impute",
    #                         action="store_true")
    # speed_group.add_argument("-e", "--exhaustive", help="exhaustive impute",
    #                          action="store_true")
    model_group.add_argument("-mi", "--mean_imputation", help="perform mean imputation (default option)",
                             action="store_true")
    model_group.add_argument("-sg", "--single_gaussian", help="impute using a single multivariate Gaussian fitted with EM",
                             action="store_true")
    model_group.add_argument("-gmm", "--gaussian_mixture", help="impute using a Gaussian mixture model fitted with EM",
                             action="store_true")
    model_group.add_argument("-dp", "--dirichlet_process", help="impute using a Dirichlet process",
                            action="store_true")
    model_group.add_argument("-mix", "--mixed_dp_gmm", help="impute using a combination of DPs and GMMs",
                            action="store_true")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-s", "--sample", help="number of samples to take from distribution",
                              type=int, default=None)
    output_group.add_argument("-m", "--mode", help="impute using the mode of the distribution (default option)",
                              action="store_true")
    parsed_args = parser.parse_args()
    
    main(parsed_args)
