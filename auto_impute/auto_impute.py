# James Allingham
# Feb 2018
# auto_impute.py
# Main file for AutoImpute CLI

import argparse
import numpy as np

import csv_reader
import mi
import sg
import gmm
import bgmm
import vigmm
import cmm
import mmm

def main(args):
    # set random seed
    if args.rand_seed: np.random.seed(args.rand_seed)

    reader = csv_reader.CSVReader(args.file, args.delimiter, args.header, args.indicator)

    data = reader.get_masked_data()

    if args.verbose:
        print("Read %s with %s rows and %s columns." % ((args.file, ) + data.shape))
        out_str = "Number missing elements: "
        for i in range(data.shape[1]):
            out_str += "col %s: %s  " % (i, np.sum(data.mask[:, i]))
        print(out_str)
        print("Percentage missing elements: %s\n" % (np.mean(data.mask),))

    if args.infinite_gmm:
        model = vigmm.VIGMM(data, args.num_components, verbose=args.verbose)
        model.fit(max_iters=args.max_iters, ϵ=args.epsilon)
    elif args.bayesian_gmm:
        model = bgmm.BGMM(data, args.num_components, verbose=args.verbose)
        model.fit(max_iters=args.max_iters, ϵ=args.epsilon)
    elif args.gaussian_mixture:
        model = gmm.GMM(data, args.num_components, verbose=args.verbose, map_est=not args.ml_estimation)
        model.fit(max_iters=args.max_iters, ϵ=args.epsilon)
    elif args.single_gaussian:
        model = sg.SingleGaussian(data, verbose=args.verbose)
        model.fit(max_iters=args.max_iters, ϵ=args.epsilon)
    elif args.categorical_mixture:
        model = cmm.CMM(data, args.num_components, verbose=args.verbose, map_est=not args.ml_estimation)
        model.fit(max_iters=args.max_iters, ϵ=args.epsilon)
    elif args.mixed_mixture:
        model = mmm.MMM(data, args.num_components, verbose=args.verbose, assignments=args.column_assignments)
        model.fit(max_iters=args.max_iters, ϵ=args.epsilon)
    else:
        model = mi.MeanImpute(data, verbose=args.verbose)

    print("")

    if args.sample:
        samples_Xs =  model.sample(args.sample)
        for s in range(args.sample):
            if args.header: 
                np.savetxt("repaired_%s.txt" % s, samples_Xs[s, :, :], delimiter=args.delimiter, header=reader.get_column_names)
            else: 
                np.savetxt("repaired_%s.txt" % s, samples_Xs[s, :, :], delimiter=args.delimiter)
    else:
        imputed_X =  model.impute()
        if args.header: 
            np.savetxt("repaired.txt", imputed_X, delimiter=args.delimiter, header=reader.get_column_names)
        else: 
            np.savetxt("repaired.txt", imputed_X, delimiter=args.delimiter)

    if args.test is not None:
        test_data = np.genfromtxt(args.test, delimiter=args.delimiter)
        imputed_X =  model.impute()
        print("RMSE: %s" % np.sqrt(np.mean(np.power(test_data - imputed_X, 2))))

    print("LL: %s" % model.log_likelihood())



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
                        type=str, default='')
    parser.add_argument("-e", "--epsilon", help="ϵ (model stopping criterion): if LL_new - LL_old < ϵ then stop iterating (default: 1e-1)",
                        type=float, default=1e-1)
    parser.add_argument("-n", "--max_iters", help="maximum number of iterations to fit model (default: 100)",
                        type=int, default=100)
    parser.add_argument("-mle", "--ml_estimation", help="use MLE rather than MAP for non-Bayesian models (default: False)",
                        action="store_true")

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
    model_group.add_argument("-bgmm", "--bayesian_gmm", help="impute using a Gaussian mixture model fitted with Variational Bayes",
                             action="store_true")
    model_group.add_argument("-vigmm", "--infinite_gmm", help="impute using a infinite Gaussian mixture model fitted with Variational Bayes",
                             action="store_true")
    model_group.add_argument("-cmm", "--categorical_mixture", help="impute using a categorical mixture model fitted with EM",
                             action="store_true")
    model_group.add_argument("-mmm", "--mixed_mixture", help="impute using a mixed (consisiting of categorical and gaussian components) mixture model fitted with EM",
                             action="store_true")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-s", "--sample", help="number of samples to take from distribution",
                              type=int, default=None)
    output_group.add_argument("-m", "--mode", help="impute using the mode of the distribution (default option)",
                              action="store_true")
    parsed_args = parser.parse_args()
    
    main(parsed_args)
