
.libPaths("/users/lforesti/rlib")
library(RcppCNPy)

fileNameBase = '/store/msrad/radar/precip_attractor/marray.big.df'

fmt = 'rda'
fileName = paste(fileNameBase,fmt, sep = ".")
load(fileName)

dataMatrix <- data.matrix(marray.big.df)
colNames <- colnames(marray.big.df)

# Save data matrix
fmt = 'npy'
fileName = paste(fileNameBase, fmt, sep = ".")
npySave(fileName, dataMatrix)

# Save varNames
fmt = 'csv'
fileName = paste(fileNameBase, 'varNames', fmt, sep = ".")
write.csv(colNames, fileName)