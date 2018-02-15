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
                        default=",")
    parser.add_argument("-ih", "--infer_header", help="use the first row as column names (default: true)",
                        default="true", type=bool)
    speed_group = parser.add_mutually_exclusive_group()
    speed_group.add_argument("-f", "--fast", help="quick impute",
                            action="store_true")
    speed_group.add_argument("-e", "--exhaustive", help="exhaustive impute",
                             action="store_true")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-s", "--sample", help="sample from distribution",
                            action="store_true")
    output_group.add_argument("-m", "--mean", help="take mean of distribution",
                            action="store_true")
    args = parser.parse_args()
    
    if args.verbose:
        print("Verbose")
    
    print(args.file)
    
    main()