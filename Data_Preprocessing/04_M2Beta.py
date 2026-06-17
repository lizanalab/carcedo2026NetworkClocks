import pandas as pd
import sys
###########
case = sys.argv[1]	#	with_Sex / with_Age / without_All
if case not in ["with_Sex", "with_Age", "without_All"]:
	print("Argument Error!")
	print("three options : with_Sex / with_Age / without_All")
	sys.exit()
###########
# ----------------------------------
file = f"./M_{case:s}.tsv"
M = pd.read_csv(file, sep="\t", header=0, index_col=0)
# -----------------------------------
beta = 2**M / (1. + 2**M)
beta.to_csv(f"./Beta_{case:s}.tsv",sep="\t",index=True, header = True)
# ---------------------------------
