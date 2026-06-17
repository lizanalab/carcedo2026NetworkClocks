import numpy as np
import pandas as pd
import sys
###########
StudyID = sys.argv[1]
num = int(sys.argv[2])
###########
# ----------------------------------
file = f"./{StudyID:s}/beta_normalized{num:d}.tsv"
beta = pd.read_csv(file, sep="\t", header=0, index_col=0)
CpGs = beta.index[:]
Samples = beta.columns[:]
#-----------------------------------
# ----------------------------------
#file = f"./Info"
#meta = pd.read_csv(file, sep="\t")
#meta = meta.set_index("ID_REF")
#meta["Age"] = pd.to_numeric(meta["Age"])
#-----------------------------------
M = np.log2(beta / (1.-beta))
M.to_csv(f"./{StudyID:s}/M{num:d}.tsv",sep="\t",index=True, header = True)

# ---------------------------------
