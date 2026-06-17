args <- commandArgs(trailingOnly = TRUE)
StudyID = args[1]
Data = args[2]	# GSE or others
num = args[3]

#-------------------------------------------------------------------------------------
outpath = paste0("./", StudyID, "/IDAT", num, "/")
Info <- read.delim(paste0("./Meta/", StudyID), header = TRUE, sep = "\t", stringsAsFactors = FALSE, colClasses = c(Sex = "character"))
rownames(Info) <- Info$"ID_REF"

#-------------------------------------------------------------------------------------
# Read Red idat files
sampsh <- list.files(path = outpath, pattern = "_Red[.]idat$", full.names = FALSE)

#
base <- sub("_Red[.]idat$", "", sampsh)

# separate "_"
parts <- do.call(rbind, strsplit(base, "_", fixed = TRUE))


if (Data=="GSE"){
# parts[,1] = GSM ID
# parts[,2] = Sentrix_ID
# parts[,3] = R01C01

samplesheet <- data.frame(
	Study_ID = StudyID,
	Sample_ID		 = parts[, 1],
	Sample_Name		 = parts[, 1],
	Sample_Group	 = "All",		   # for QC
	Sentrix_ID		 = parts[, 2],
	Sentrix_Position = parts[, 3],
	stringsAsFactors = FALSE)
} else{	# E_MTAB
# parts[,1] = Sentrix_ID
# parts[,2] = R0XC0X

samplesheet <- data.frame(
	Study_ID = StudyID,
	Sample_ID		 = paste(parts[, 1], parts[,2], sep="_"),
	Sample_Name		 = paste(parts[, 1], parts[,2], sep="_"),
	Sample_Group	 = "All",		   # for QC
	Sentrix_ID		 = parts[, 1],
	Sentrix_Position = parts[, 2],
	stringsAsFactors = FALSE)
}

rownames(samplesheet) <- samplesheet$"Sample_ID"
#-------------------------------------------------------------------------------------
drop_cols <- c("ID_REF")
Info_sub <- Info[, setdiff(colnames(Info), drop_cols), drop = FALSE]

common <- intersect(rownames(samplesheet), rownames(Info_sub))
common <- common[match(common, rownames(samplesheet))]

out <- cbind(samplesheet[common, , drop = FALSE], Info_sub[common, , drop = FALSE])

#-------------------------------------------------------------------------------------
# write CSV
write.table(
  out,
  file = paste0("./", StudyID, "/samplesheet",num,".csv"),
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
