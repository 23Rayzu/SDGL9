#===========================================================
# ACARA 9 - Feature Selection (Sains Data Geospasial Lanjutan)
#===========================================================

#--- Atur working directory ---
setwd("D:/SIG MUKHLISH/Semester 5/SDGL/ACARA 9/ACARA 9")

#--- Library yang diperlukan ---
library(rsample)
library(Boruta)
library(caret)
library(corrplot)
library(ggplot2)
library(randomForest)

#===========================================================
# BAGIAN 1: Feature Selection dengan Boruta
#===========================================================

#--- Load data training dari folder BAHAN ---
input.table <- read.csv("BAHAN/acara03b_bahan.csv")
cat("Data berhasil dimuat. Jumlah kolom:", ncol(input.table), "\n")
head(input.table)

# Gunakan kolom ke-2 sampai ke-104 sesuai kebutuhan analisis
input.table <- input.table[, 2:104]

#--- Cek korelasi antar data ---
cortes <- cor(input.table)
col <- colorRampPalette(c("red", "white", "blue"))(20)

# Plot dan simpan hasil korelasi
dev.new(bg = "white")
corrplot(cortes, type = "upper", order = "hclust", col = col)
dev.copy(png, "OUTPUT/plot_korelasi.png", width = 1000, height = 800, bg = "white")
dev.off()
cat("plot_korelasi.png berhasil disimpan di folder OUTPUT\n")

#--- Split data training dengan perbandingan 70:30 ---
df.frame <- initial_split(input.table, prop = 0.7)
train2 <- training(df.frame)
test2 <- testing(df.frame)
cat("Data berhasil dibagi menjadi training dan testing set\n")

#--- Lakukan variable selection menggunakan Boruta ---
cat("Proses Boruta sedang berjalan...\n")
v.selection <- Boruta(factor(LU) ~ ., data = train2, maxRuns = 25, doTrace = 2)

#--- Plot hasil Boruta ---
dev.new(bg = "white")
par(bg = "white")
plot(v.selection, main = "Hasil Seleksi Fitur dengan Boruta")
dev.copy(png, "OUTPUT/plot_boruta.png", width = 1000, height = 800, bg = "white")
dev.off()
cat("plot_boruta.png berhasil disimpan di folder OUTPUT\n")

#--- Turunkan variable yang fix ---
v.selection.fix <- TentativeRoughFix(v.selection)

#--- Ambil formula hasil Boruta ---
formula.boruta <- getConfirmedFormula(v.selection.fix)
cat("Variabel yang dipilih oleh Boruta:\n")
print(formula.boruta)

#===========================================================
# BAGIAN 2: Recursive Feature Elimination (RFE)
#===========================================================

#--- Atur kontrol untuk RFE ---
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 5,
                      number = 5,
                      verbose = TRUE)

#--- Pisahkan data target dan variabel prediktor ---
y.train <- factor(train2$LU)
x.train <- train2[, 2:103]

#--- Jalankan RFE ---
cat("Proses Recursive Feature Elimination (RFE) sedang berjalan...\n")
result_rfe1 <- rfe(x = x.train, y = y.train,
                   sizes = c(1:13),
                   rfeControl = control)

#--- Lihat variabel terpilih ---
cat("Variabel yang dipilih oleh RFE:\n")
print(predictors(result_rfe1))

#--- Plot hasil RFE ---
dev.new(bg = "white")
ggplot(data = result_rfe1, metric = "Accuracy") + 
  theme_bw() +
  ggtitle("Akurasi Berdasarkan Jumlah Fitur (RFE)")
dev.copy(png, "OUTPUT/plot_rfe.png", width = 1000, height = 800, bg = "white")
dev.off()
cat("plot_rfe.png berhasil disimpan di folder OUTPUT\n")

#--- Importance variable dari hasil RFE ---
cat("Importance dari variabel hasil RFE:\n")
print(varImp(result_rfe1))

#===========================================================
# SIMPAN HASIL UNTUK PRAKTIKUM BERIKUTNYA
#===========================================================

save.image("PROJECT/feature_selection_result.RData")
cat("Hasil analisis disimpan sebagai: feature_selection_result.RData\n")

