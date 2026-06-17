import pandas as pd
import sys
###########
var = sys.argv[1]	#	Sex / Age / Without
if var == "Sex":
	case = "with_Sex"
elif var == "Age":
	case = "with_Age"
else:
	case = "without_All"
###########
# ----------------------------------
file = f"./M_{case:s}.tsv"
M = pd.read_csv(file, sep="\t", header=0, index_col=0)
# -----------------------------------
beta = 2**M / (1. + 2**M)
beta.to_csv(f"./Beta_{case:s}.tsv",sep="\t",index=True, header = True)
# ---------------------------------
