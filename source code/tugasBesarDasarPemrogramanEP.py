import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates

file_path = '/home/domidavid/CodingFIles/Dasar Pemrograman/Data/smart_grid_dataset.csv'  # ubah sesuai lokasi file Anda
data = pd.read_csv(file_path)

# Mengelompokkan kolom FFT
fft_columns = [col for col in data.columns if col.startswith('FFT_')]

# Buat daftar pilihan untuk pengguna
options = []

# Tambahkan pilihan FFT 
for fft_col in fft_columns:
    options.append(fft_col)

# Tambahkan kolom selain Timestamp dan FFT ke daftar pilihan
other_columns = [col for col in data.columns if col not in fft_columns + ['Timestamp']]
options.extend(other_columns)

# Tampilkan daftar pilihan
print("Pilih kolom sinyal yang ingin dianalisis dari daftar berikut:")
for i, option in enumerate(options, start=1):
    print(f"{i}. {option}")

# Input opsi dari pengguna dengan validasi
while True:
    try:
        choice = int(input(f"Masukkan nomor pilihan (1-{len(options)}): "))
        if 1 <= choice <= len(options):
            selected_column = options[choice - 1]
            break
        else:
            print(f"Masukkan angka antara 1 sampai {len(options)}.")
    except ValueError:
        print("Input tidak valid. Masukkan nomor angka.")

print(f"Anda memilih kolom: {selected_column}")

# Ambil kolom waktu dan sinyal yang dipilih
x = pd.to_datetime(data['Timestamp'])
y = data[selected_column].values.reshape(-1, 1)

# Normalisasi data
scaler = MinMaxScaler()
y_norm = scaler.fit_transform(y).flatten()

# Noise removal dengan median filter pada data normalisasi
kernel_size = 5 if len(y_norm) > 5 else 3
y_denoised = medfilt(y_norm, kernel_size=kernel_size)

# Smoothing dengan Savitzky-Golay filter pada data hasil noise removal
window_length = 51 if len(y_denoised) > 51 else (len(y_denoised)//2)*2 + 1
polyorder = 3
y_smoothed = savgol_filter(y_denoised, window_length=window_length, polyorder=polyorder)

# Menampilkan grafik
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(x, y_smoothed, color='purple', label='Data setelah Normalisasi, Noise Removal, dan Smoothing')
ax.set_title('Hasil Pengolahan Sinyal Smart Grid')
ax.set_xlabel('Timestamp')
ax.set_ylabel(selected_column)  
ax.grid(True)
ax.legend()

# Format tanggal dan waktu agar jelas dan mudah dibaca
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

