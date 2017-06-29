#!/usr/bin/env Rscript

.libPaths("/users/lforesti/rlib")
library(RcppCNPy)

###
scaleKM="016"
fileNameBase = paste('marray.big.df.',scaleKM, sep="")
dirNameIn = '/store/msrad/radar/precip_attractor/maple_data/r-database/'
fileNameBaseIn = paste(dirNameIn, fileNameBase, sep="")
dirNameOut = '/store/msrad/radar/precip_attractor/maple_data/python-database/'
fileNameBaseOut = paste(dirNameOut, fileNameBase, sep="")###

fmt = 'rda'
fileName = paste(fileNameBaseIn,fmt, sep = ".")

print('Reading R database...')
load(fileName)
print('R database read.')

dataMatrix <- data.matrix(marray.big.df)
colNames <- colnames(marray.big.df)

# Split data matrix into two parts
size <- nrow(dataMatrix)

# Save data matrix in python format
fmt = 'npy'

print('Writing python database...')
if (size < 150000){
    fileName = paste(fileNameBaseOut, fmt, sep = ".")
    npySave(fileName, dataMatrix)
} else {
    fileName = paste(fileNameBaseOut,'p1', fmt, sep = ".")
    npySave(fileName, dataMatrix[0:int(size/2),:])
    fileName = paste(fileNameBaseOut,'p2', fmt, sep = ".")
    npySave(fileName, dataMatrix[int(size/2)+1:size,:])
}
print('Python database written.')

# Save varNames
fmt = 'csv'
fileName = paste(fileNameBaseOut, 'varNames', fmt, sep = ".")
write.csv(colNames, fileName)
