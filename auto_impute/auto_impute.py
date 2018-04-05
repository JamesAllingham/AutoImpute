# James Allingham
# Feb 2018
# auto_impute.py
# Main file for AutoImpute CLI

import argparse
import numpy as np

import csv_reader
import sg
import gmm
import mi

def main(args):
    # set random seed
    if args.rand_seed: np.random.seed(args.rand_seed)

    reader = csv_reader.CSVReader(args.file, args.delimiter, args.header, args.indicator)

    data = reader.get_masked_data()

    if args.verbose:
        print("Read %s with %s rows and %s columns." % ((args.file, ) + data.shape))
        out_str = "Number missing elements: "
        for i in range(data.shape[1]):
            out_str += "col %s: %s  " % (i, np.sum(np.isnan(data[:, i])))
        print(out_str)
        print("Percentage missing elements: %s\n" % (np.mean(np.isnan(data)),))

    if args.gaussian_mixture:
        model = gmm.GMM(data, 3, verbose=args.verbose)
        model.fit()
    elif args.single_gaussian:
        model = sg.SingleGaussian(data, verbose=args.verbose)
        model.fit()
    else:
        model = mi.MeanImpute(data, verbose=args.verbose)

    if args.test is not None:
        test_data = np.genfromtxt(args.test, delimiter=args.delimiter)
        # imputed_X = model.sample(1)
        imputed_X =  model.impute()
        # print(imputed_X)
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
                        type=str,default=None)
    parser.add_argument("-i", "--indicator", help="inidcator string that a value is missing (default: '')",
                        type=str,default='')

    model_group = parser.add_mutually_exclusive_group()
    # speed_group.add_argument("-f", "--fast", help="quick impute",
    #                         action="store_true")
    # speed_group.add_argument("-e", "--exhaustive", help="exhaustive impute",
    #                          action="store_true")
    model_group.add_argument("-mi", "--mean_imputation", help="perform mean imputation",
                             action="store_true")
    model_group.add_argument("-sg", "--single_gaussian", help="impute using a single multivariate Gaussian fitted with EM",
                             action="store_true")
    model_group.add_argument("-gmm", "--gaussian_mixture", help="impute using a Gaussian mixture model fitted with EM",
                             action="store_true")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-s", "--sample", help="number of samples to take from distribution",
                              type=int, default=None)
    output_group.add_argument("-mm", "--mean_mode", help="take mean or mode of continuous or discrete distributions, respectively (default option)",
                              action="store_true")
    parsed_args = parser.parse_args()
    
    main(parsed_args)
