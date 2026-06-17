library(data.table)
library(sva)
library(BiocParallel)

args <- commandArgs(trailingOnly = TRUE)
var = args[1]	#	Age, Sex, or Without

#
CHUNK_SIZE <- 10000 # 10K,  5000 ~ 50000
#

register(SerialParam())

#--------------------------
# Dataset definitions: M-value file, samplesheet (Study_ID, Sample_ID, Age, Sex), CellComposition
datasets <- list(
	list(m = "./E_MTAB_4931/M1.tsv",		ss = "./E_MTAB_4931/samplesheet1.csv",			cc = "./E_MTAB_4931/cellProp1.tsv"),
	list(m = "./GSE51032/M1.tsv",			ss = "./GSE51032/samplesheet1.csv",				cc = "./GSE51032/cellProp1.tsv"),
	list(m = "./GSE61496/M1.tsv",			ss = "./GSE61496/samplesheet1.csv",				cc = "./GSE61496/cellProp1.tsv"),
	list(m = "./GSE81961/M1.tsv",			ss = "./GSE81961/samplesheet1.csv",				cc = "./GSE81961/cellProp1.tsv"),
	list(m = "./GSE87640/M1.tsv",			ss = "./GSE87640/samplesheet1.csv",				cc = "./GSE87640/cellProp1.tsv"),
	list(m = "./GSE87648/M1.tsv",			ss = "./GSE87648/samplesheet1.csv",				cc = "./GSE87648/cellProp1.tsv"),
	list(m = "./GSE99624/M1.tsv",			ss = "./GSE99624/samplesheet1.csv",				cc = "./GSE99624/cellProp1.tsv"),
	list(m = "./GSE107737/M1.tsv",			ss = "./GSE107737/samplesheet1.csv",			cc = "./GSE107737/cellProp1.tsv"),
	list(m = "./GSE125105/M1.tsv",			ss = "./GSE125105/samplesheet1.csv",			cc = "./GSE125105/cellProp1.tsv"),
	list(m = "./GSE42861/M1.tsv",			ss = "./GSE42861/samplesheet1.csv",				cc = "./GSE42861/cellProp1.tsv"),
	list(m = "./GSE59065/M1.tsv",			ss = "./GSE59065/samplesheet1.csv",				cc = "./GSE59065/cellProp1.tsv"),
#
	list(m = "./GSE87571/M1.tsv",			ss = "./GSE87571/samplesheet1.csv",				cc = "./GSE87571/cellProp1.tsv"),
	list(m = "./GSE87571/M2.tsv",			ss = "./GSE87571/samplesheet2.csv",				cc = "./GSE87571/cellProp2.tsv"),
	list(m = "./GSE87571/M3.tsv",			ss = "./GSE87571/samplesheet3.csv",				cc = "./GSE87571/cellProp3.tsv")
)

# Output
if(var == "Sex"){
	out_all <- "M_with_Sex.tsv"
	out_sample <- "Samplesheet_with_Sex.csv"
} else if(var == "Age"){
	out_all <- "M_with_Age.tsv"
	out_sample <- "Samplesheet_with_Age.csv"
} else{
	out_all <- "M_without_All.tsv"
	out_sample <- "Samplesheet_without_All.csv"
}

#--------------------------
# Helpers
clean_id <- function(x) trimws(as.character(x))

read_mmat <- function(path) {
	dt <- fread(path, sep = "\t", data.table = FALSE, check.names = FALSE)
	cpg <- dt[[1]]
	dt <- dt[, -1, drop = FALSE]
	mat <- as.matrix(dt)
	rownames(mat) <- clean_id(cpg)
	colnames(mat) <- clean_id(colnames(mat))
	storage.mode(mat) <- "double"
	mat
}

# Read CSV/TSV with auto separator
read_table_any <- function(path) {
	if (!file.exists(path)) return(NULL)
	tryCatch(fread(path, data.table = FALSE, check.names = FALSE),
		error = function(e) NULL)
}

# Find a column by candidate names (case-insensitive)
find_col <- function(df, candidates) {
	if (is.null(df)) return(NA_character_)
	nm <- names(df)
	for (cand in candidates) {
		hit <- which(tolower(nm) == tolower(cand))
		if (length(hit) > 0) return(nm[hit[1]])
	}
	NA_character_
}

# Standardize Sex values to factor F/M
# Handles "M"/"F", "Male"/"Female", "1"/"2", lowercase, with whitespace etc.
standardize_sex <- function(x) {
	x <- toupper(trimws(as.character(x)))
	out <- rep(NA_character_, length(x))
	out[x %in% c("MALE", "M", "1")] <- "M"
	out[x %in% c("FEMALE", "F", "2")] <- "F"
	factor(out, levels = c("F", "M"))
}

ComBat_write_by_rows <- function(dat, batch, mod = NULL, chunk_size = 20000, out_all,
	par.prior = TRUE, prior.plots = FALSE) {
	stopifnot(is.matrix(dat))
	stopifnot(length(batch) == ncol(dat))
	nr <- nrow(dat)
	if (file.exists(out_all)) file.remove(out_all)

	first <- TRUE
	for (start in seq(1, nr, by = chunk_size)) {
		end <- min(start + chunk_size - 1, nr)
		idx <- start:end

		cb <- ComBat(dat = dat[idx, , drop = FALSE], batch = batch, mod = mod,
			par.prior = par.prior, prior.plots = prior.plots)

		dt <- data.table::as.data.table(cb)
		dt[, CpG := rownames(cb)]
		data.table::setcolorder(dt, c("CpG", setdiff(names(dt), "CpG")))

		data.table::fwrite(dt, out_all, sep = "\t", append = !first, col.names = first)

		first <- FALSE
		rm(cb, dt, idx); gc()
	}
}

# ---------------------------------------------------------
# Step 1: Read samplesheets (Study_ID + Sample_ID + Age + Sex)
meta_list <- lapply(datasets, function(d){
	md <- read.table(d$ss, sep = ",", header = TRUE, stringsAsFactors = FALSE, check.names = FALSE, colClasses = c(Sex="character"))
	md$Sample_ID <- paste(md$Study_ID, trimws(as.character(md$Sample_ID)), sep="__")
	md$Sex <- as.character(md$Sex)
	md$Age <- as.numeric(md$Age)
	md
})
merged_meta <- data.table::rbindlist(meta_list, fill = TRUE, use.names = TRUE)
merged_meta <- as.data.frame(merged_meta)
write.csv(merged_meta, out_sample, row.names = FALSE)



# Identify Age and Sex columns from merged metadata
#	Age
age_col <- find_col(merged_meta, c("Age"))

cat("=== Detected covariate columns ===\n")
cat("Age column:", age_col, "\n")

#	Sex
sex_col <- find_col(merged_meta, c("Sex", "sex", "Gender", "gender", "SEX", "GENDER"))

if (is.na(sex_col)) stop("No Sex/Gender column in samplesheets. Available: ",
	paste(names(merged_meta), collapse = ", "))

cat("=== Detected covariate columns ===\n")
cat("Sex column:", sex_col, "\n")



# Build phenotype frame from samplesheet
merged_phen <- data.frame(
	Sample_ID = merged_meta$Sample_ID,
	Sex = standardize_sex(merged_meta[[sex_col]]),
	Age = as.numeric(merged_meta[[age_col]]),
	stringsAsFactors = FALSE
)
merged_phen <- merged_phen[!duplicated(merged_phen$Sample_ID), , drop = FALSE]


# ---------------------------------------------------------
# Step 3: Read M-values and build batch
m_list <- mapply(function(d, md) {
		mat <- read_mmat(d$m)
		samps <- colnames(mat)
		original_id <- sub("^.*?__", "", md$Sample_ID)
		study_id <- md$Study_ID[match(samps, original_id)]
		colnames(mat) <- paste(study_id, samps, sep = "__")
		mat
	}, datasets, meta_list, SIMPLIFY = FALSE)

# Common CpGs
common_cpg <- Reduce(intersect, lapply(m_list, rownames))
if (length(common_cpg) < 1000) stop("Common CpGs too few: ", length(common_cpg))
m_list <- lapply(m_list, function(m) m[common_cpg, , drop = FALSE])

M <- do.call(cbind, m_list)

batch_raw <- sub("__.*$", "", unlist(lapply(m_list, colnames)))
batch <- factor(batch_raw)

cat("=== batch levels ===\n")
print(table(batch, useNA = "always"))

# ---------------------------------------------------------
# Step 4: Build covariate matrix (Sex)

# Verify coverage
miss_phen <- setdiff(colnames(M), merged_phen$Sample_ID)
if (length(miss_phen) > 0) {
	stop("Samples missing from samplesheet:\n",
		paste(head(miss_phen, 10), collapse = "\n"),
		if (length(miss_phen) > 10) "\n..." else "")
}


# Reorder phen and cc to match M columns
phen2 <- merged_phen[match(colnames(M), merged_phen$Sample_ID), , drop = FALSE]
stopifnot(identical(phen2$Sample_ID, colnames(M)))


if(var == "Sex"){
	mod <- model.matrix(~ Sex, data = phen2)

	cat("=== Covariates: Sex ===\n")
	cat("=== mod dim:", paste(dim(mod), collapse = " x "), "===\n")
	cat("=== Sex distribution ===\n")
	print(table(phen2$Sex, useNA = "always"))

	stopifnot(nrow(mod) == ncol(M))

} else if(var == "Age"){
	mod <- model.matrix(~ Age, data = phen2)

	cat("=== mod dim:", paste(dim(mod), collapse = " x "), "===\n")
	stopifnot(nrow(mod) == ncol(M))
} else{
	mod <- NULL
}
# ---------------------------------------------------------
# Step 5: Run ComBat
ComBat_write_by_rows(dat = M, batch = batch, mod = mod, chunk_size = CHUNK_SIZE,
	out_all = out_all, par.prior = TRUE, prior.plots = FALSE)

cat("Done. Output: ", out_all, "\n", sep = "")

# ---------------------------------------------------------
# Step 6: Diagnostics
cat("\n=== M-value sample count ===\n"); print(ncol(M))
cat("=== Samplesheet sample count ===\n"); print(nrow(merged_meta))

cat("=== Samples in M but not in samplesheet ===\n")
in_M_not_meta <- setdiff(colnames(M), merged_meta$Sample_ID)
print(length(in_M_not_meta)); print(head(in_M_not_meta))

