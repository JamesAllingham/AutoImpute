import argparse

def main():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically repairs CSV files with missing entries")
    parser.add_argument("file", help="name of the file to repair")
    parser.add_argument("-v", "--verbose", help="increase the output verbosity",
                        action="store_true")
    parser.add_argument("-d", "--delimiter", help="file delimiter (default: ',')",
                        type=str,default=",")
    parser.add_argument("-hd", "--header", help="use the first row as column names",
                        action="store_true")
    parser.add_argument("-rs", "--rand_seed", help="random seed to use (default: 42)",
                        type=int, default=42)

    model_group = parser.add_mutually_exclusive_group()
    # speed_group.add_argument("-f", "--fast", help="quick impute",
    #                         action="store_true")
    # speed_group.add_argument("-e", "--exhaustive", help="exhaustive impute",
    #                          action="store_true")
    model_group.add_argument("-mi", "--mean_imputation", help="perform mean imputation",
                        action="store_false")
    model_group.add_argument("-emg", "--em_gaussian", help="impute using a multivariate Gaussian fitted with EM",
                        action="store_false")
    model_group.add_argument("-emgmm", "--em_gaussian_mixture", help="impute using a Gaussian mixture model fitted with EM",
                        action="store_false")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-s", "--sample", help="number of samples to take from distribution",
                            type=int)
    output_group.add_argument("-mm", "--mean_mode", help="take mean or mode of continuous or discrete distributions, respectively (default option)",
                            action="store_true")
    args = parser.parse_args()
    
    if args.verbose:
        print("Verbose")
    
    print(args.file)
    
    main()