# Laporan Proyek Machine Learning 

Nama:  Aprillia Nur Azizah 

Gmail: aprilliaazizah3@gmail.com

## 1. Domain Proyek 
Stroke merupakan penyakit gangguan fungsional otak akut lokal maupun global akibat terhambatnya aliran darah ke otak karena pendarahan (Stroke Hemoragik) ataupun sumbatan (Stroke Iskemik) (Junaidi., 2011). Menurut data Yayasan Stroke Indonesia (Yastroki), masalah stroke di Indonesia semakin genting dan mendesak, karena jumlah penderita stroke di Indonesia menduduki urutan pertama di Asia. Jumlah yang disebabkan oleh penyakit stroke menduduki urutan kedua pada usia di atas 60 tahun dan urutan kelima pada usia 15-59 tahun (Yastroki., 2012). 

Dengan banyaknya jumlah kasus stroke ini maka dari itu diperlukan prediksi dini untuk memastikan apakah seorang pasien berpotensi stroke atau tidak. Dimana klasifikasi ini nantinya dapat digunakan sebagai prediksi dan juga penanganan dini pasien yang dapat mengurangi jumlah penderita stroke di Indonesia. 

(Sailasya & Aruna Kumari, n.d.) Membuat sebuah model prediksi penyakit stroke dengan menggunakan algoritma seperti Logistic Regression, Decision Tree, Random Forest, KNN, SVM, dan Naive Bayes yang menghasilkan nilai akurasi sebesar 77.5%, 77.5%, 72%, 77.4%, 78.6%, dan 79.2%. Dimana algoritma Naive Bayes memiliki nilai akurasi terbaik jika dibandingkan dengan Algoritma lainnya (https://thesai.org/Downloads/Volume12No6/Paper_62Analyzing_the_Performance_of_Stroke_Prediction.pdf). 

## 2. Business Understanding 
### 2.1 Problem Statements 
Bagaimana cara mengklasifikasi otomatis penyakit stroke? 
### 2.2 Goals
Untuk mempercepat waktu diagnosis pada pasien yang terindikasi terkena penyakit stroke sehingga dapat mengurangi resiko kematian dan kecelakaan kerja akibat penyakit stroke. 
### 2.3 Solution Statements 
Membuat model yang dapat melakukan klasifikasi dan prediksi dini apakah seseorang terinfeksi penyakit stroke dengan menggunakan model Machine Learning yaitu Logistic Regression. Logistic Regression merupakan bagian dari analisis regresi yang digunakan ketika variabel dependen (respon) merupakan variabel dikotomi. Variabel dikotomi terdiri dari dua nilai yang mewakili kemunculan atau tidak adanya suatu kejadian, biasanya diberi angka 0 atau 1 (Saputro & Widodo, 2014) . Model Logistic Regression merupakan model paling efektif dan efisien dalam memprediksi sesuatu, hanya saja model ini rentan terhadap *underfitting* yang jumlah kelasnya tidak seimbang (Rianto et al., 2015). 

## 3. Data Understanding 
Dataset yang digunakan pada submission ini adalah *Stroke Prediction Dataset* (https://www.kaggle.com/fedesoriano/stroke-prediction-dataset/tasks?taskId=4888). Pada dataset ini terdapat 12 *column* yang terdiri dari 5110 sample data. Variabel-variabel yang terdapat pada *Stroke Prediction Dataset* adalah sebagai berikut:
- id = nomor identitas pasien. 
- gender = jenis kelamin pasien. 
- age = umur pasien.
- hypertension = menunjukkan apakah seorang pasien memiliki penyakit hipertensi dengan nilai 0 berarti tidak memiliki dan 1 berarti memiliki.
- heart disease =  menunjukkan apakah seorang pasien memiliki penyakit jantung dengan nilai 0 berarti tidak memiliki dan 1 berarti memiliki.
- ever married = menunjukkan apakah seorang pasien sudah menikah atau belum.
- work type = menjelaskan tipe pekerjaan seorang pasien. 
- residence type = menjelaskan tipe tempat kerja seorang pasien. 
- avg glucose level = menunjukkan nilai rata-rata level glukosa seorang pasien 
- bmi = menunjukkan nilai body massa index seorang pasien. 
- smoking status = menjelaskan apakah pasien seorang perokok atau tidak. 
- stroke = menunjukkan apakah seorang pasien terkena stroke atau tidak dengan nilai 0 (tidak stroke) dan 1(stroke).

### 3.1. Data Visualization
#### 3.1.1 Korelasi antar atribut 
![Alt Text](https://raw.githubusercontent.com/aprilliaazizah/picture/main/korelasi.png) 

Berdasarkan gambar diatas dapat dilihat bahwa atribut *age* memiliki nilai korelasi yang cukup kuat dengan stroke dibandingkan atribut lainnya. 

#### 3.1.2 Menampilkan korelasi antara penyakit Hypertension dengan penyakit Stroke
![Alt Text](https://raw.githubusercontent.com/aprilliaazizah/picture/main/hypertensi.png)

Berdasarkan gambar diatas dapat diketahui bahwa seseorang yang memiliki penyakit Hypertension lebih rentan terkena stroke dibandingkan yang tidak.  

#### 3.1.3 Melihat Korelasi antara Perokok dengan penyakit Stroke
![Alt Text](https://raw.githubusercontent.com/aprilliaazizah/picture/main/smoking.png)

Berdasarkan gambar diatas dapat diketahui bahwa orang yang terkena stroke lebih didominasi oleh orang yang pernah merokok dari pada yang tidak.  

### 3.1.4 Melihat Korelasi antara jenis kelamin dengan penyakit Stroke
![Alt Text](https://raw.githubusercontent.com/aprilliaazizah/picture/main/gender.png) 

Berdasarkan gambar diatas dapat diketahui bahwa orang yang terkena stroke didominasi oleh pria.  

#### 3.1.5 Melihat Korelasi antara tempat tinggal dengan penyakit Stroke
![Alt Text](https://raw.githubusercontent.com/aprilliaazizah/picture/main/resindence.png)

Berdasarkan gambar diatas dapat diketahui bahwa orang yang terkena stroke didominasi oleh orang yang tinggal di daerah perkotaan.


Dapat dilihat bahwa penyakit hipertensi, seorang perokok, seseorang berjenis kelamin laki-laki, dan tinggal di perkotaan memiliki tingkat korelasi yang tinggi dengan stroke. Dimana hal-hal tersebut dapat menjadi penyebab terjadinya penyakit stroke. 



## 4. Data Preparation 
### 4.1. Data Cleaning 
Data Cleaning merupakan proses menyiapkan data untuk dilakukan analisis dengan cara menghapus atau memodifikasi data salah, tidak relevan, duplikat, dan tidak terformat. Pada tahapan ini proses data cleaning yang dilakukan ialah:
1. Menghapus column yang tidak dibutuhkan, seperti variable id.
2. Mengidentifikasi dan menghapus *missing value* serta *duplicate data*, fungsinya agar tidak terdapat data yang tidak tersedia dan data yang memiliki nilai yang sama.
3. Mengindentifikasi outliers, fungsinya sendiri adalah untuk menghindari bias yaitu dengan cara mengambil data yang nilainya terlalu jauh dengan data lainnya kemudian dihapus.

### 4.2. Feature Selection 
Feature Selection merupakan suatu proses yang bertujuan untuk memilih feature yang berpengaruh dan mengesampingkan feature yang tidak berpengaruh dalam suatu kegiatan pemodelan atau penganalisaan data. Pada tahapan ini atribut yang akan menjadi target ialah variable stroke, dimana variable ini akan dipisahkan dengan atribut lainnya. Untuk mengurangi ketidakseimbangan kelas pada atribut yang menjadi target maka digunakanlah SMOTE (*Synthetic Minority Over-sampling Technique*). SMOTE sendiri merupakan jenis augmentasi data yang mensintesis sample baru dari sampel yang sudah ada sehingga kelas pada atribut yang menjadi target menjadi seimbang.
Setelah data selesai diolah barulah kemudian data dibagi menjadi 80% data training dan 20% data testing. Berdasarkan hasil rasio tersebut, dihasilkan data training sebanyak 5358 sampel data, dan data testing sebanyak 1340 sampel data.

## 5. Modeling 
Data yang sudah dibagi kemudian di proses pada tahapan ini untuk menghasilkan sebuah prediksi dengan menggunakan algoritma Logistic Regression. Logistic Regression merupakan salah satu metode statistika yang sering digunakan untuk menganalisis data yang mendeskripsikan antara variabel respon dengan satu atau lebih variabel prediksi (Bimantara & Dina, 2019). Variabel respon dari Logistic Regression bersifat dikotomi yang hanya bernilai 1 (ya) dan 0 (tidak) (Hosmer et al., 2013). Dalam hal ini data yang telah diproses model akan memberikan keluaran apakah seseorang tersebut terkena stroke atau tidak. Hasil akurasi yang didapatkan dengan menggunakan algoritma Logistic Regression dengan parameter *random_state* = 0 adalah 79% untuk data train dan 81% untuk data test.

![Alt Text](https://raw.githubusercontent.com/aprilliaazizah/picture/main/logistic.PNG)


## 6. Evaluation
Pada tahapan ini, saya menggunakan metode *Confusion Matrix* untuk mengukur kinerja model yang sudah dibuat, dengan membandingkan dan menghitung nilai 4 kombinasi *Confusion Matrix* yaitu TP (*True Positive*), FP (*False Positive*), FN (*False Negative*), dan TN (*True Negative*) untuk menentukkan nilai:
![Alt Text](https://raw.githubusercontent.com/aprilliaazizah/picture/main/confusion_matrix.PNG)

Hasil yang didapatkan pada penelitian ini adalah sebagai berikut.

![Classification Report Logistic Regression](https://raw.githubusercontent.com/aprilliaazizah/picture/main/hasil_cm1.PNG)

![Plot Confusion Matrix Logistic Regression](https://raw.githubusercontent.com/aprilliaazizah/picture/main/hasil_cm2.PNG)

Pada Gambar diatas didapatkan hasil klasifikasi yakni True Positive (TP) = 505, True Negative (TN) = 507, False Positive (FP) = 106, False Negative (FN) = 142. Sehingga dapat disimpulkan bahwa model ini dapat memprediksi dengan cukup baik. Hasil akurasi yang didapatkan juga sebesar 81%. 


## Daftar Referensi
Junaidi, I. 2011. Stroke Waspadai Ancamannya. Penerbit Andi, Yogyakarta.

Yayasan Stroke Indonesia. 2012. Angka Kejadian Stroke Meningkat Tajam. http://www.yastroki.or.id/ read.php? id=317.

Rianto, H., Tinggi, S., Informatika, M., Komputer, D., & Mandiri, N. (2015). Resampling Logistic Regression untuk Penanganan Ketidakseimbangan Class pada Prediksi Cacat Software. Journal of Software Engineering, 1(1). http://journal.ilmukomputer.org

Sailasya, G., & Aruna Kumari, G. L. (n.d.). Analyzing the Performance of Stroke PModelingrediction using ML Classification Algorithms. IJACSA) International Journal of Advanced Computer Science and Applications, 12(6), 2021. Retrieved November 9, 2021, from www.ijacsa.thesai.org

Saputro, R. A., & Widodo, P. P. (2014). KOMPARASI ALGORITMA KLASIFIKASI DATA MINING UNTUK MEMPREDIKSI PENYAKIT TUBERCULOSIS (TB): STUDI KASUS PUSKESMAS KARAWANG SUKABUMI. SNIT 2014, 1(1), 120–126. http://seminar.bsi.ac.id/snit/index.php/snit-2014/article/view/188

Bimantara, A., & Dina, T. A. (2019). Klasifikasi Web Berbahaya Menggunakan Metode Logistic Regression. Annual Research Seminar (ARS), 4(1), 173–177. https://seminar.ilkom.unsri.ac.id/index.php/ars/article/view/1932

Hosmer, D. W., Lemeshow, S., Sturdivant, R. X., & Army Academy, U. S. (2013). Applied Logistic Regression Third Edition. www.wiley.com.















