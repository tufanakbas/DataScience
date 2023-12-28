import pandas as pd
from collections import Counter

def calculate_word_frequencies(input_csv, output_csv):
    # CSV dosyasını oku
    df = pd.read_csv(input_csv)

    # 'İçerik' sütunundaki tüm kelimeleri birleştir ve virgülle ayır
    all_text = ','.join(df['İçerik'].astype(str))
    words = all_text.split(',')

    # Kelime frekanslarını hesapla
    word_frequencies = Counter(words)

    # Frekansları içeren bir DataFrame oluştur
    frequency_df = pd.DataFrame(list(word_frequencies.items()), columns=['Kelime', 'Frekans'])

    # Frekans sütununa göre büyükten küçüğe sırala
    frequency_df = frequency_df.sort_values(by='Frekans', ascending=False)

    # CSV dosyasına yaz
    frequency_df.to_csv(output_csv, index=False)

    print(f"Kelimelerin frekansları '{output_csv}' dosyasına yazıldı.")

# Kullanılacak CSV dosyalarının adlarını ve yollarını güncelleyin
input_csv_path = 'output_4.csv'
output_csv_path = 'word_frequencies1.csv'

# Fonksiyonu çağır
calculate_word_frequencies(input_csv_path, output_csv_path)
