library(missForest)

#data = read.csv("./Git/AutoImpute/investigations/test_MAR.csv", header = FALSE, na.strings = "nan")
#data.true = read.csv("./Git/AutoImpute/data/boston-0-MCAR.csv", header = FALSE)
#summary(data)
#data.imp <- missForest(data, xtrue = data.true, verbose = TRUE)

folder <- "./Git/AutoImpute/data/report/"

for (name in c("boston", "iris")){
  for (mistype in c("NMAR", "MAR", "MCAR")){
    for (perc in c("10", "20", "30", "40", "50")){
      for (i in c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")){
        file_name <- paste0(name, "_", perc, "_", mistype, "_",  i, ".csv")
        print(file_name)
        data = read.csv(paste0(folder, file_name), header = FALSE, na.strings = "nan")
        data.imp <- missForest(data)
        write.table(data.imp$ximp, file = paste0(folder, name, "_", perc, "_", mistype, "_",  i, "_mf.csv"), sep = ",",row.names=FALSE, col.names = FALSE)
      }
    }
  }
}

