# Install CRAN packages needed for the assignment
cran <- c(
  "data.table",  # fast IO / data ops
  "dplyr",       # tidy ops
  "readr",       # read_csv
  "pROC"         # AUC
)
# choose a fast, reliable CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages(cran, dependencies = TRUE)
