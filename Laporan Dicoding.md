# Laporan Proyek Machine Learning 
#
#### Nama   : Ramdani Tarjianto
#### Email  : ramdani.tarjianto83@gmail.com
#
#
#
#
#


# Judul
##### Mendeteksi Penyakit Jantung dengan Algortima Neural Network dan Membandingkan Beberapa Algoritma Klasifikasi (Machine Learning)
#
#
#
## 1.DOMAIN PROYEK

###### 1.1 LATAR BELAKANG
Heart Disease, juga dikenal sebagai penyakit cardiovascular diseases (CVD) adalah penyebab kematian nomor 1 di dunia. Penyakit cardiovascular diseases (CVD) adalah gangguan pada jantung dan pembuluh darah termasuk penyakit jantung koroner, penyakit serebrovaskular, gagal jantung, dan jenis patologi lainnya. Secara keseluruhan, penyakit cardiovascular menyebabkan kematian sekitar 17 juta orang di seluruh dunia setiap tahun, dengan angka kematian meningkat untuk pertama kalinya dalam 50 tahun di Inggris (Chicco and Jurman, 2020). Karena masih kurangnya kesadaran akan gaya hidup sehat dan kurangnya informasi tentang penyakit jantung, yang mungkin membuat gejala awal tidak dapat dikenali. Proses deteksi penyakit jantung dapat dilakukan secara manual yaitu konsultasi langsung dengan ahli jantung dan beberapa pemeriksaan laboratorium, kemudian ahli jantung harus berkonsultasi kembali. Tentunya hal ini membutuhkan biaya yang relatif besar. Karena tingginya risiko kematian, maka diperlukan suatu sistem yang dapat mendeteksi penyakit jantung pada pasien secara akurat dengan harga yang murah (Wibisono and Fahrurozi, 2019). 

###### 1.2 MASALAH DALAM DOMAIN
Cara menerapkan algoritma machine learning untuk memprediksi pasien dengan penyakit jantung agar dapat memberikan perhatian yang lebih kepada pasien tersebut dan dapat meningkatkan pelayanan kesehatan yang lebih baik.

###### 1.3 HASIL RISET TERKAIT
Terdapat penelitian yang bertujuan untuk memprediksi pasien dengan penyakit jantung menggunakan algoritma C4.5  (http://research.pps.dinus.ac.id/index.php/Cyberku/article/view/4/4).
#
#
#
#
#
## 2.BUSINESS UNDERSTANDING
###### 2.1 PROBLEM STATEMENTS
Mendeteksi Penyakit Jantung dengan Algortima Neural Network dan Membandingkan Beberapa Algoritma Klasifikasi Machine Learning
###### 2.2 TUJUAN
- Melakukan evaluasi model prediksi penyakit jantung.
- Mengetahui faktor-faktor penyakit atau variabel bebas seperti gula darah, denyut jantung, kolesterol, dan lain-lain yang mempengaruhi terjadinya penyakit jantung.
- Membuat model penyakit jantung menggunakan algortima machine learning.

###### 2.3 MANFAAT 
- Memprediksi seseorang yang berpotensi penyakit jantung secara akurat yang berguna untuk meningkatkan pelayanan terhadap pasien tersebut.
- Membantu staf medis memprediksi apakah seorang pasien terkena penyakit jantung dengan menggunakan algoritma machine learing
- Memberitahu masyarakat luas tentang bagaimana cara bekerja dalam memprediksi penyakit jantung menggunakan algoritma machine learning

###### 2.4 SOLUTION STATEMENTS
Saya mengajukan algoritma machine learning sebagai solusi permasalahan, yaitu Neural Network dan membadingkannya dengan Logistic Regression, SGD, dan Naive Bayes.

Metode Neural Network (NN) (Derisma, 2020), Naive Bayes (Klasifikasi, Putra and Rini, 2019) dan Logistic Regression (Kumar and Devi Gandhi, 2018) diusulkan oleh banyak peneliti untuk prediksi penyakit jantung.

Neural Network memiliki kelebihan dalam mengambil keputusan dengan memprediksi data. Karena masukan yang digunakan dalam memprediksi penyakit lebih banyak dan diagnosis harus dilakukan pada tahap yang berbeda, Neural Network memperluas kemampuan prediktifnya pada tingkat hierarki yang berbeda dalam struktur jaringan berlapis-lapis. Struktur berlapis-lapis ini membantu dalam memilih fitur dari kumpulan data pada skala yang berbeda untuk menyempurnakannya menjadi fitur yang lebih spesifik (Boddu and Subhadra, 2019). Neural Network mempunyai kelemahan yaitu permasalahan dalam penentuan parameter sehingga perlu dilakukan eksperimen dalam menentukan tiap parameternya (Somantri and Cilacap, 2018). 

Naive Bayes akurasi yang dihasilkan cukup baik, hal ini karena keunggulan dari metode naive bayes sendiri yaitu mampu melakukan klasifikasi meskipun memiliki data training yang sedikit untuk estimasi parameternya (Devita, Wahyu Herwanto and Wibawa, 2018). Namun Naive Bayes memiliki kelemahan yaitu atribut atau fitur independen sering salah dan hasil estimasi probabilitas tidak dapat berjalan optimal (Prabowo and Muljono, 2018).

Logistic Regression sangat berguna untuk memprediksi ada atau tidaknya karakteristik atau hasil berdasarkan nilai dari set variabel prediktor (Putra and Rini, 2020) yang mana ini tepat untuk menentukan pasien dalam hal terkena penyakit jantung atau tidak. Tetapi Logistic Regression memiliki kelemahan yaitu rentan terhadap underfitting pada dataset yang kelasnya tidak seimbang, sehingga akan menghasilkan akurasi yang rendah (Rianto and Wahono, 2015).
#
#
#
#
#
## 3. DATA UNDERSTANDING
Pada dataset yang digunakan pada proyek akhir ini adalah heart.csv.(https://www.kaggle.com/johnsmith88/heart-disease-dataset)
Dataset ini adalah untuk memprediksi apakah seseorang akan mengidap penyakit jantung atau tidak menggunakan teknik pembelajaran mesin. Dataset ini merupakan data nyata termasuk fitur penting pasien. Ada beberapa fitur yang harus dipahami dengan baik untuk memastikan atau mendapatkan akurasi dengan baik, Terdapat 1025 sampel data dan 14 column yang digunakan pada dataset ini, column tersebut diantaranya terdiri dari:
 - age = merupakan umur pasien
 - sex = merupakan jenis kelamin pasien
 - chest pain type  = merupakan jenis nyeri dada pada pasien
 - resting blood = merupakan jumlah resting blood pada pasien
 - serum cholesterol = merupakan jumlah kolesterol pada pasien
 - fasting blood sugar = merupakan jumlah gula darah pada pasien
 - resting electrocardiographic = merupakan jumlah elektrokardiografi pada pasein
 - maximum heart rate achieved = merupakan jumlah detak jantung maksimal pada pasien
 - exercise induced angina = merupakan jumlah agina pada pasein
 - oldpeak = merupakan jumlah oldpeak pada pasien
 - slope = merupakan jumlah slope pada pasien
 - cha = merupakan jumlah cha pada pasien
 - thal = merupakan jumlah thal pada pasien
 - target = merupakan nilai yang akan di prediksi 


Nilai yang akan diprediksi yaitu kolom target yang berisi nilai positif dan negatif, ini akan diprediksi menggunakan algoritma klasifikasi Neural Network yang akan mengeluarkan output 0 atau 1. 0 artinya pasien tersebut tidak terkena penyakit jantung dan 1 terkena penyakit jantung berdasarkan dengan input dari 14 column tersebut. Dataset ini diambil dari kaggle
(https://www.kaggle.com/johnsmith88/heart-disease-dataset)
#
#
#
#
#
## 4. DATA PREPARATION
Melakukan Data Preprocessing pada dataset heart disease (‘hear.csv’): dengan Memeriksa Missing Values, Kegunaan metode ini adalah agar di dalam dataset tidak ada nilai yang hilang dan agar tidak mempengaruhi hasil prediksi akan tetapi tidak terdapat missing values pada dataset tersebut yang artinya dataset sudah bisa langsung di bagi menjadi train dan test menggunakan train_test_split.
![Data Preparations](https://raw.githubusercontent.com/RamdaniTarjianto/Dicoding/main/Screenshot%20(182).png)
jadi data sudah bisa langsung di bagi menjadi train dan test menggunakan train_test_split
#
#
#
#
#
## 5. MODELING
![](https://raw.githubusercontent.com/RamdaniTarjianto/Dicoding/main/Screenshot%20(183).png)
untuk mendeteksi pasein dengan penyakit jantung saya menggunakn algortima Neural Network karena Neural Network memiliki kelebihan dalam mengambil keputusan dengan memprediksi data. Karena masukan yang digunakan dalam memprediksi penyakit lebih banyak dan diagnosis harus dilakukan pada tahap yang berbeda, Neural Network memperluas kemampuan prediktifnya pada tingkat hierarki yang berbeda dalam struktur jaringan berlapis-lapis. Struktur berlapis-lapis ini membantu dalam memilih fitur dari kumpulan data pada skala yang berbeda untuk menyempurnakannya menjadi fitur yang lebih spesifik (Boddu and Subhadra, 2019)

Setelah data sudah di split menggunakan train_test_split dari sklearn.model_selection data sudah siap untuk masuk ke tahap modeling.

untuk Neural Network saya mengimport MLPClassifier dari sklearn.neural_network, MLPClassifier sendiri adalah singkatan dari Multi-layer Perceptron classifier yang dalam namanya terhubung ke Neural Network. Tidak seperti algoritme klasifikasi lain seperti Support Vectors Machine atau Naive Bayes Classifier, MLPClassifier mengandalkan Neural Network yang mendasari untuk melakukan tugas klasifikasi. dan saya menggunakan parameter random_state = 2  agar hasil prediksi tidak berubah-ubah setiap kali running ulang.

untuk algortima SGD saya mengimport SGDClassifier dengan random state = 2, kemudian import accuracy_score untuk mengetahui hasil akurasi dari model tersebut.

untuk algortima Logistic Regression saya mengimport LogisticRegression dengan random state = 2, kemudian import accuracy_score untuk mengetahui hasil akurasi dari model tersebut.

untuk algortima Naive Bayes saya mengimport GaussianNB. kemudian import accuracy_score untuk mengetahui hasil akurasi dari model tersebut.

#
#
#
#
#
## 6. EVALUATION
Confusion matrix juga sering disebut error matrix. Pada dasarnya confusion matrix memberikan informasi perbandingan hasil klasifikasi yang dilakukan oleh sistem (model) dengan hasil klasifikasi sebenarnya. Confusion matrix berbentuk tabel matriks yang menggambarkan kinerja model klasifikasi pada serangkaian data uji yang nilai sebenarnya diketahui. Gambar dibawah ini merupakan confusion matrix dengan 4 kombinasi nilai prediksi dan nilai aktual yang berbeda. Perhatikan gambar dibawah ini:
![](https://raw.githubusercontent.com/RamdaniTarjianto/Dicoding/main/Screenshot%20(184).png)
Pada gambar tersebut menjelaskan, dengan menggunakan confusion matrix kita akan mendapatkan 4 sebagai representasi hasil proses klasifikasi yakni True Positive (TP) =  91, True Negative (TN) = 75, False Positive (FP) = 24, False Negative (FN) = 15. Nilai True Positif (TP) merupakan data positif yang terdeteksi dengan benar, sedangkan nilai True Negative (TN) merupakan data negatif yang terdeteksi dengan benar. False Positive (FP) merupakan data positif yang terdeteksi dengan salah, sedangkan False Negative (FN) merupakan data negatif yang terdeteksi dengan salah.

Selain akurasi saya juga menggunakan classification report yang menampilkan metrik Presisi, Recall, f1-score, Support. dan mendapatkan Presisi = 0.83 untuk variable 0(orang yang tidak terkena penyakit jantung) dan 0.79 untuk variable 1(orang yang terkena penyakit jantung), Recal = 0.76 untuk variable 0(orang yang tidak terkena penyakit jantung) dan 0.86 untuk variable 1(orang yang terkena penyakit jantung), f1-score = 0.79 untuk variable 0(orang yang tidak terkena penyakit jantung) dan 0.82 untuk variable 1(orang yang terkena penyakit jantung).
    - Presisi = merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan hasil yang diprediksi positf.
    - Recal = Merupakan rasio prediksi benar positif dibandingkan dengan keseluruhan data yang benar positif.
    - f1-score = F1 Score merupakan perbandingan rata-rata presisi dan recall yang dibobotkan

Algoritma diatas Logistic Regression, Naive Bayes, SGD dan Neural Network menghasilakn nilai akurasi sebagai berikut :
| No | Algoritma | Nilai Akurasi |
| --- | --- | --- |
| 1 | Neural Network | 0.81 |
| 2 | Stochastic Gradient Descent | 0.59 |
| 3 | Logistic Regression | 0.8 |
| 4 | Naive Bayes | 0.8 |
#
#
#
#
## KESIMPULAN
Penelitian ini membuktikan bahwa Klasifikasi menggunakan  Algoritma Neural Network dalam mendeteksi penyakit jantung menghasilkan Akurasi sebesar 81% , Presisi sebesar 79%, Recall sebesar 86%, dan F1 score sebesar 82%. Penelitian ini dapat disimpulkan bahwa Neural Network mempunyai keunggulan untuk memprediksi ada atau tidaknya karakteristik atau hasil berdasarkan nilai dari set variabel prediktor yang mana ini tepat untuk mendeteksi pasien dengan penyakit jantung atau tidak.
#
#
#
#
##  DAFTAR PUSTAKA

Alhamad, A. et al. (2019) ‘Prediksi Penyakit Jantung Menggunakan Metode-Metode Machine Learning Berbasis Ensemble – Weighted Vote’, Jurnal Edukasi dan Penelitian Informatika (JEPIN), 5(3), p. 352. doi: 10.26418/jp.v5i3.37188.

Chicco, D. and Jurman, G. (2020) ‘Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone’, BMC Medical Informatics and Decision Making, 20(1), p. 16. doi: 10.1186/s12911-020-1023-5.

Devita, R. N., Wahyu Herwanto, H. and Wibawa, A. P. (2018) ‘PERBANDINGAN KINERJA METODE NAIVE BAYES DAN K-NEAREST NEIGHBOR UNTUK KLASIFIKASI ARTIKEL BERBAHASA INDONESIA PERFORMANCE COMPARISON OF NAIVE BAYES AND K-NEAREST NEIGHBOR METHODS FOR INDONESIAN ARTICLES CLASSIFICATION’, 5(4), pp. 427–434. doi: 10.25126/jtiik.201854773.

Klasifikasi, A., Putra, P. D. and Rini, D. P. (2019) Annual Research Seminar (ARS), Prosiding Annual Research Seminar. Available at: http://archive.ics.uci.edu/ml/machine-learning-databases/ (Accessed: 28 April 2021).

Prabowo, A. D. R. and Muljono, M. (2018) ‘Prediksi Nasabah Yang Berpotensi Membuka Simpanan Deposito Menggunakan Naive Bayes Berbasis Particle Swarm Optimization’, Techno.Com, 17(2), pp. 208–219. doi: 10.33633/tc.v17i2.1648.

Putri, I. E., Rahmawati, D. and Yufis Azhar, ; (2020) ‘COMPARISON OF DATA MINING CLASSIFICATION METHODS TO DETECT HEART DISEASE’, Jurnal Pilar Nusa Mandiri, 16(2), pp. 213–218. doi: 10.33480/pilar.v16i2.1481.

View of Perbandingan Kinerja Algoritma untuk Prediksi Penyakit Jantung dengan Teknik Data Mining (no date). Available at: https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/2152/1142 (Accessed: 28 April 2021).

Zhang, Y., Diao, L. and Ma, L. (2021) ‘Logistic Regression Models in Predicting Heart Disease’, Journal of Physics: Conference Series, 1769, p. 12024. doi: 10.1088/1742-6596/1769/1/012024.

Shipe, M. E. et al. (2019) ‘Developing prediction models for clinical use using logistic regression: An overview’, Journal of Thoracic Disease, 11(Suppl 4), pp. S574–S584. doi: 10.21037/jtd.2019.01.25.

Boddu, V. and Subhadra, K. (no date) Neural network based intelligent system for predicting heart disease, International Journal of Innovative Technology and Exploring Engineering (IJITEE). Available at: https://www.researchgate.net/publication/332035370 (Accessed: 29 April 2021).
Wibisono, A. B. and Fahrurozi, A. (2019) ‘Perbandingan Algoritma Klasifikasi Dalam Pengklasifikasian Data Penyakit Jantung Koroner’, Jurnal Ilmiah Teknologi dan Rekayasa, 24(3), pp. 161–170. doi: 10.35760/tr.2019.v24i3.2393.

Arsi, P. and Somantri, O. (2018) ‘Deteksi Dini Penyakit Diabetes Menggunakan Algoritma Neural Network Berbasiskan Algoritma Genetika’, Jurnal Informatika: Jurnal Pengembangan IT, 3(3), pp. 290–294. doi: 10.30591/jpit.v3i3.1008.