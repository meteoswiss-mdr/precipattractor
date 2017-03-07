
readRZC <- function(filename){
    
    data_list <- read_metranet_file(filename)
    precipIndex <- producePrecipIndex("RZC")
    precip <- transformRadarImageToPrecip(data_list$data, precipIndex)    
    }

read_metranet_file <- function(radar_file,debug.mode=FALSE){


     #
     # R function to read METRANET radar products
     #
     # require external shared library 
     #
     # Return a "list"
     #
     # data : matrix of data 
     # conv_table: to convert data in to real unit
     # prd_header: metadata of the product
     #
     # Example: 
     #
     #    radar_file<- '/home/lom/users/mbc/tmp/RZC161311200VL.801'
     #    radar_file<- './TMP/RZC1406300107L.801'
     #    a<-read_metranet_file(radar_file,metranet.lib = metranet.lib)
     #
     #    names(a)
     #  [1] "data"       "conv_table" "prd_header"
     #
     #

     # History
     # -------
     # 20160512 mbc first release
     # 20160919 mbc "intelligent" detect Linux/Solaris environment
     #               DN 0 for RZC/CZC/EZC => NA
     #
    
    apply_spatial_filter <- FALSE
    
    metranet.path        <-"/store/mch/msrad/idl/lib/radlib4/"      #Path of metranet libraries
    metranet.lib         <- c( "srn_idl_py_lib.x86_64.so",
                               "srn_idl_py_lib.sparc64.so",
                               "srn_idl_py_lib.sparc32.so")         #Metranet library names  
     
     metranet_lib_linux <- metranet.lib[[1]]
     metranet_lib_sparc64 <- metranet.lib[[2]]
     metranet_lib_sparc32 <- metranet.lib[[3]]
     
     
     system_info <- as.list(Sys.info())

     if (grepl("Sun", system_info$sysname) ){
           metranet_lib <- metranet_lib_sparc32
     } else if (grepl("Linux", system_info$sysname) ){
           metranet_lib <- metranet_lib_linux
     }
     dyn.load(paste(metranet.path,metranet_lib,sep=""))
     #print(a)
     
     if(debug.mode) print(getLoadedDLLs()[length(getLoadedDLLs())])

     #Find where the last dot is located in the metranet_lib
     m.ind = gregexpr(pattern =".",metranet_lib,fixed=T)
     m.ind=m.ind[[1]][length(m.ind[[1]])]-1
     metranet.package = substring(metranet_lib,1,m.ind)
     if(debug.mode) print(metranet.package)


     out <- NA
     
     # default values
     row <- 640
     column <- 710
     table_level <- 256

     # special mean for DN '0'
     special_prdt <- c("RZC", "CZC", "EZC")

     radar_files <- Sys.glob(radar_file)
     nr_files <- length(radar_files)
     
     if ( nr_files > 0 ) {
           radar_file <- radar_files[nr_files]
           
           # read ASCII part
           #Here small warning. Report to mbc
           #ascii_header <- read.table(radar_file, header = FALSE, sep="=", as.is=TRUE, fill=TRUE,skipNul = TRUE )
           suppressWarnings(ascii_header <- read.table(radar_file, header = FALSE, sep="=", as.is=TRUE, fill=TRUE))
           j <- which (ascii_header == "end_header")
           if ( j > 0 ) {
                ascii_header <- ascii_header[1:j-1,]
                prd_header <- ascii_header[,2]
                names(prd_header) <- ascii_header[,1]

                row <- as.numeric(prd_header["row"])
                column <- as.numeric(prd_header["column"])
           } else {
                prd_header <- NA
           }

           data_size <- row * column

           if (data_size > 0 ) {
           ret <- .C("r_decoder", file=as.character(radar_file), dataptr=as.raw(rep(0,data_size)), n_elements=as.integer(data_size), prd_table=as.single(rep(0,table_level)), NAOK=FALSE, PACKAGE=metranet.package)

                data <- matrix(as.integer(ret$dataptr), column, row)

                if (apply_spatial_filter)
                     data=filter_data(data)

                conv_table <- ret$prd_table

                # convert 0 at end of array with NA
                i <- length(conv_table)
                if (i > 0) conv_zero2NA <- TRUE
                else conv_zero2NA <- FALSE

                while (conv_zero2NA) {
                     if (conv_table[i] == 0) {
                           conv_table[i] <- NA
                     } else {
                           conv_zero2NA <- FALSE
                     }
                     i <- i-1
                     if ( i <= 0) conv_zero2NA <- FALSE
                }

                # special mean for DN '0'
                if ( is.element(prd_header["pid"], special_prdt) ) {
                     conv_table[1] <- NA
                }

                out <- list(data, conv_table, prd_header)
                names(out) <- c("data", "conv_table", "prd_header")
           }
     }
     return(out)
}

producePrecipIndex <- function( type ,precip.index.factor=71.5) {
     
     precip.index = vector(length=256)
     if(type == "AQC") {  
           #print("AQC")
           precip.index[1] = 0
           precip.index[2] = 0
           precip.index[251:256] = 0       
           for(i in 2: 250) precip.index[[i+1]] = (10.^((i-(precip.index.factor))/20.0)/316.0)^(0.6666667) 
     }
     
     if(type == "RZC" | type=="RZF") {    
           #print("RZC")
           precip.index = 
           c(0,0.000000e+00,3.526497e-02,7.177341e-02,1.095694e-01,1.486983e-01,
                1.892071e-01,2.311444e-01,2.745606e-01,3.195080e-01,3.660402e-01,
                4.142135e-01,4.640857e-01,5.157166e-01,5.691682e-01,6.245048e-01,
                6.817929e-01,7.411011e-01,8.025010e-01,8.660660e-01,9.318726e-01,
                1.000000e+00,1.070530e+00,1.143547e+00,1.219139e+00,1.297397e+00,
                1.378414e+00,1.462289e+00,1.549121e+00,1.639016e+00,1.732080e+00,
                1.828427e+00,1.928171e+00,2.031433e+00,2.138336e+00,2.249010e+00,
                2.363586e+00,2.482202e+00,2.605002e+00,2.732132e+00,2.863745e+00,
                3.000000e+00,3.141060e+00,3.287094e+00,3.438278e+00,3.594793e+00,
                3.756828e+00,3.924578e+00,4.098242e+00,4.278032e+00,4.464161e+00,
                4.656854e+00,4.856343e+00,5.062866e+00,5.276673e+00,5.498019e+00,
                5.727171e+00,5.964405e+00,6.210004e+00,6.464264e+00,6.727490e+00,
                7.000000e+00,7.282120e+00,7.574187e+00,7.876555e+00,8.189587e+00,
                8.513657e+00,8.849155e+00,9.196485e+00,9.556064e+00,9.928322e+00,
                1.031371e+01,1.071269e+01,1.112573e+01,1.155335e+01,1.199604e+01,
                1.245434e+01,1.292881e+01,1.342001e+01,1.392853e+01,1.445498e+01,
                1.500000e+01,1.556424e+01,1.614837e+01,1.675311e+01,1.737917e+01,
                1.802731e+01,1.869831e+01,1.939297e+01,2.011213e+01,2.085664e+01,
                2.162742e+01,2.242537e+01,2.325146e+01,2.410669e+01,2.499208e+01,
                2.590869e+01,2.685762e+01,2.784002e+01,2.885706e+01,2.990996e+01,
                3.100000e+01,3.212848e+01,3.329675e+01,3.450622e+01,3.575835e+01,
                3.705463e+01,3.839662e+01,3.978594e+01,4.122425e+01,4.271329e+01,
                4.425483e+01,4.585074e+01,4.750293e+01,4.921338e+01,5.098415e+01,
                5.281737e+01,5.471524e+01,5.668003e+01,5.871411e+01,6.081992e+01,
                6.300000e+01,6.525696e+01,6.759350e+01,7.001244e+01,7.251669e+01,
                7.510925e+01,7.779324e+01,8.057188e+01,8.344851e+01,8.642657e+01,
                8.950967e+01,9.270148e+01,9.600586e+01,9.942677e+01,1.029683e+02,
                1.066347e+02,1.104305e+02,1.143601e+02,1.184282e+02,1.226398e+02,
                1.270000e+02,1.315139e+02,1.361870e+02,1.410249e+02,1.460334e+02,
                1.512185e+02,1.565865e+02,1.621438e+02,1.678970e+02,1.738531e+02,
                1.800193e+02,1.864030e+02,1.930117e+02,1.998535e+02,2.069366e+02,
                2.142695e+02,2.218609e+02,2.297201e+02,2.378564e+02,2.462797e+02,
                2.550000e+02,2.640278e+02,2.733740e+02,2.830498e+02,2.930668e+02,
                3.034370e+02,3.141730e+02,3.252875e+02,3.367940e+02,3.487063e+02,
                3.610387e+02,3.738059e+02,3.870234e+02,4.007071e+02,4.148732e+02,
                4.295390e+02,4.447219e+02,4.604402e+02,4.767129e+02,4.935594e+02,
                5.110000e+02,5.290557e+02,5.477480e+02,5.670995e+02,5.871335e+02,
                6.078740e+02,6.293459e+02,6.515750e+02,6.745881e+02,6.984126e+02,
                7.230773e+02,7.486119e+02,7.750469e+02,8.024141e+02,8.307465e+02,
                8.600779e+02,8.904438e+02,9.218805e+02,9.544258e+02,9.881188e+02,
                1.023000e+03,1.059111e+03,1.096496e+03,1.135199e+03,1.175267e+03,
                1.216748e+03,1.259692e+03,1.304150e+03,1.350176e+03,1.397825e+03,
                1.447155e+03,1.498224e+03,1.551094e+03,1.605828e+03,1.662493e+03,
                1.721156e+03,1.781888e+03,1.844761e+03,1.909852e+03,1.977238e+03,
                2.047000e+03,2.119223e+03,2.193992e+03,2.271398e+03,2.351534e+03,
                2.434496e+03,2.520384e+03,2.609300e+03,2.701352e+03,2.796650e+03,
                2.895309e+03,2.997448e+03,3.103188e+03,3.212656e+03,3.325986e+03,
                3.443312e+03,3.564775e+03,3.690522e+03,3.820703e+03,3.955475e+03,
                4.095000e+03,4.239445e+03,4.388984e+03,4.543796e+03,4.704068e+03,
                4.869992e+03,5.041768e+03,5.219600e+03,5.403705e+03,5.594301e+03,
                0,          0,          0,          0,          0
                             )
     }
     
     
     return(precip.index)
}

transformRadarImageToPrecip <- function(image,precip.index,pixel.values=FALSE) {

     #Check if all the necessary arguments have been given.     
     if(missing(image))              stop("Please specify the value to the argument 'image' to proceed.")
     if(missing(precip.index)) stop("Please specify the value to the argument 'precip.index' to proceed.")

     precip<-array(0,dim=dim(image))
     #image<-as.integer(round(image))
     image<-floor(image)
     
     #Here it is probably more correct to place 
     #precip[,]<-(precip.index[image[,]+1]+precip.index[image[,]+2])/2
     #or in log scale
     #precip[,]<- 10^(0.5*log10(precip.index[image[,]+1]*precip.index[image[,]+2]))
     #or
     #precip[,]<- precip.index[  (image[,]+ (image[,]+1) )/2    + 1]
     #So we assign precip values at the middle of the class rather at the bottom limit
     
     #precip[,]<-precip.index[image[,]+1]
     precip[,] <- 10^(0.5*log10(precip.index[image[,]+1]*precip.index[image[,]+2]))
     precip[which(image==255)]<-NA
     
     if(pixel.values==TRUE) precip<-image
     
     #attr(precip,"filename")<-attributes(image)$filename
     
     return(precip)
}


