import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="parse unit size data in preprocee_data"
    )
    arg_parser.add_argument(
        dest="filename",
        help="filename"
    )
    args = arg_parser.parse_args()

    f = open(args.filename,'r')
    lines = f.readlines()
    units = np.array([float(line.split(':')[-1]) for line in lines])
    
    sns.distplot(units)
    plt.show()