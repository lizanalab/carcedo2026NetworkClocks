library(ChAMP)
library(minfi)
library(FlowSorted.Blood.EPIC)

args <- commandArgs(trailingOnly = TRUE)
StudyID = args[1]
Data = args[2]	#	GSE or others
outpath = paste0("./", StudyID, "/IDAT",num,"/")
num = args[3]
Array = args[4]	# 450K or EPIC



#-------------------------------------------------------------------------------------
#	Load (minfi method -> rgSet, mset)
rgSet <- read.metharray.exp(outpath)

ss <- read.csv(paste0("./", StudyID, "samplesheet", num,".csv"), check.names = FALSE, stringsAsFactors = FALSE)

#-------------------------------------------------------------------------------------

# sampleNames
rgSet_names <- sampleNames(rgSet)
if(Data == "GSE"){ rgSet_names_clean <- sub("_.*", "", rgSet_names) } else { rgSet_names_clean <- rgSet_names }

ss_ordered <- ss[match(rgSet_names_clean, ss$Sample_Name), ]

#-------------------------------------------------------------------------------------

# CellType -> Labelling
ct <- ss_ordered$CellType
ss_ordered$CellType <- NA
ss_ordered$CellType[ct %in% c("WB", "BC", "PB", "PBLeu")] <- "WholeBlood"
ss_ordered$CellType[ct %in% c("PBLymp", "PBMC")]          <- "PBMC"

# 
if (any(is.na(ss_ordered$CellType))) {
    warning("Unmapped CellType: ",
            paste(unique(ct[is.na(ss_ordered$CellType)]), collapse = ", "))
}


#-------------------------------------------------------------------------------------

#	Cell composition estimation
estimate_cc <- function(rgSet_subset, sample_type) {
    celltypes <- c("CD8T", "CD4T", "NK", "Bcell", "Mono")
    
    cc <- estimateCellCounts2(rgSet_subset,
                              compositeCellType = "Blood",
                              processMethod = "preprocessNoob",
                              cellTypes = celltypes
                              )
    return(as.data.frame(cc$prop))
}

#-------------------------------------------------------------------------------------
#
wb_idx <- which(ss_ordered$CellType == "WholeBlood")
pbmc_idx <- which(ss_ordered$CellType == "PBMC")
#
#-------------------------------------------------------------------------------------
#
cellProp_list <- list()

if (length(wb_idx) > 0) {
    cc_wb <- estimate_cc(rgSet[, wb_idx], "WholeBlood")
    cc_wb$Sample_Name <- ss_ordered$Sample_Name[wb_idx]
    cc_wb$CellType <- "WholeBlood"
    cellProp_list[["WB"]] <- cc_wb
}

if (length(pbmc_idx) > 0) {
    cc_pbmc <- estimate_cc(rgSet[, pbmc_idx], "PBMC")
    cc_pbmc$Sample_Name <- ss_ordered$Sample_Name[pbmc_idx]
    cc_pbmc$CellType <- "PBMC"
    cellProp_list[["PBMC"]] <- cc_pbmc
}

#
all_cols <- c("Sample_Name", "CellType", "CD8T", "CD4T", "NK", "Bcell", "Mono")
cellProp_all <- do.call(rbind, lapply(cellProp_list, function(x) {
    x[, intersect(all_cols, colnames(x)), drop = FALSE]
}))

write.table(cellProp_all,
            file = paste0("./", StudyID, "/cellProp", num,".tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)

#-------------------------------------------------------------------------------------
#	Noob (Background + Dye bias Corrections)
mset_noob <- preprocessNoob(rgSet)
beta_noob <- getBeta(mset_noob)

if(Data == "GSE"){ colnames(beta_noob) <- sub("_.*", "", colnames(beta_noob))	}

#-------------------------------------------------------------------------------------
#	Match Data Sample Names
idx <- match(colnames(beta_noob), ss$Sample_Name)
colnames(beta_noob) <- ss$Sample_Name[idx]

#-------------------------------------------------------------------------------------
#	ChAMP filter
detP <- detectionP(rgSet)

if(Data == "GSE"){ colnames(detP) <- sub("_.*", "", colnames(detP))	}

colnames(detP) <- ss$Sample_Name[idx]
myFilt <- champ.filter(beta = beta_noob,
					   pd = ss,
					   detP = detP,
					   arraytype = Arry)

#-------------------------------------------------------------------------------------
#	BMIQ
myNorm <- champ.norm(beta = myFilt$beta,
		plotBMIQ = FALSE,
        method = "BMIQ",
        arraytype = Array,
        cores = 5)

#-------------------------------------------------------------------------------------
# Write
write.table(
        cbind(CpG = rownames(myNorm), myNorm),
        file = paste0("./", StudyID, "/beta_normalized", num, ".tsv"),
        sep = "\t",
        row.names = FALSE,
        quote = FALSE)
#-------------------------------------------------------------------------------------
