import os
import csv
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder


# NLTK kütüphanesini kullanarak Türkçe stopwords listesini yükle
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("turkish"))


def create_output_folder():
    # Kodun çalıştığı dizinde bir 'output_folder' oluştur
    output_folder = 'output_folder'
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def preprocess_text(text):
    # Noktalama işaretlerini ve rakamları kaldır
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # Küçük harfe çevir
    text = text.lower()

    # Türkçe stopwords'leri kaldır
    words = word_tokenize(text, language="turkish")
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # Kelimeleri virgülle ayırarak birleştir
    preprocessed_text = " ".join(filtered_words)

    return preprocessed_text

def txt_to_csv(root_folder, output_folder):
    # Alt klasörlerin ve txt dosyalarının olduğu tüm dosyaları al
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        if os.path.isdir(folder_path):  # Eğer dosya bir klasörse devam et
            # Alt klasör adına uygun CSV dosyasını oluştur
            csv_file_path = os.path.join(output_folder, f"{folder_name}.csv")

            # CSV dosyasını aç ve yazma modunda başlat
            with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file)

                # Başlık satırını yaz
                csv_writer.writerow(["Dosya Adı", "İçerik"])

                # Her alt klasördeki txt dosyalarını oku ve CSV'ye yaz
                for txt_file_name in os.listdir(folder_path):
                    if txt_file_name.endswith(".txt"):
                        txt_file_path = os.path.join(folder_path, txt_file_name)
                        with open(txt_file_path, "r", encoding="utf-8") as txt_file_content:
                            content = txt_file_content.read()
                            preprocessed_content = preprocess_text(content)
                            csv_writer.writerow([txt_file_name, preprocessed_content])


def remove_single_letter_words(input_file, output_file, target_column):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_input, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_output:

        csv_reader = csv.DictReader(csv_input)
        fieldnames = csv_reader.fieldnames

        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()

        for row in csv_reader:
            words = row[target_column].split()
            updated_words = [word for word in words if len(word) > 1]
            row[target_column] = " ".join(updated_words)
            csv_writer.writerow(row)

    return output_file

def remove_two_letter_words(input_file, output_file, target_column):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_input, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_output:

        csv_reader = csv.DictReader(csv_input)
        fieldnames = csv_reader.fieldnames

        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()

        for row in csv_reader:
            words = row[target_column].split()
            updated_words = [word for word in words if len(word) != 2]
            row[target_column] = " ".join(updated_words)
            csv_writer.writerow(row)

    return output_file

def remove_specific_words(input_file, output_file, target_column):
    words_to_remove = ["bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on","a'nî", "ama", "amma", "ancak", "belki", "bile", "çünkü", "da", "de", "dahi", "demek", "dışında", "eğer", "encami", "fakat", "gâh", "gelgelelim", "gibi", "hâlbuki", "hatta", "hem", "ile", "ille velakin", "ille velâkin", "imdi", "kâh", "kaldı ki", "karşın", "ki", "lakin", "madem", "mademki", "maydamı", "meğerki", "meğerse", "ne var ki", "neyse", "oysa", "oysaki", "ve", "velakin", "velev", "velhâsıl", "velhâsılıkelâm", "veya", "veyahut", "ya da", "yahut", "yalıňız", "yalnız", "yani", "yok", "yoksa", "zira","göre", "kadar"]  # Silmek istediğiniz kelimelerin listesi
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_input, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_output:

        csv_reader = csv.DictReader(csv_input)
        fieldnames = csv_reader.fieldnames

        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()

        for row in csv_reader:
            words = row[target_column].split()
            updated_words = [word for word in words if word not in words_to_remove]
            row[target_column] = " ".join(updated_words)
            csv_writer.writerow(row)

    return output_file

def separate_words_with_comma(input_file, output_file, target_column):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_input, \
         open(output_file, 'w', newline='', encoding='utf-8') as csv_output:

        csv_reader = csv.DictReader(csv_input)
        fieldnames = csv_reader.fieldnames

        csv_writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
        csv_writer.writeheader()

        for row in csv_reader:
            words = row[target_column].split()
            row[target_column] = ",".join(words)
            csv_writer.writerow(row)

    return output_file

def process_csv_file(input_file, output_folder):
    # remove_single_letter_words fonksiyonu uygula
    output_file_1 = remove_single_letter_words(input_file, os.path.join(output_folder, 'output_1.csv'), 'İçerik')

    # remove_two_letter_words fonksiyonu uygula
    output_file_2 = remove_two_letter_words(output_file_1, os.path.join(output_folder, 'output_2.csv'), 'İçerik')

    # remove_specific_words fonksiyonu uygula
    output_file_3 = remove_specific_words(output_file_2, os.path.join(output_folder, 'output_3.csv'), 'İçerik')

    # separate_words_with_comma fonksiyonu uygula
    output_file_4 = separate_words_with_comma(output_file_3, os.path.join(output_folder, 'output_4.csv'), 'İçerik')

    return output_file_4

def apply_fp_growth(input_csv, output_csv):
    # CSV dosyasını oku
    df = pd.read_csv(input_csv)

    # Her bir veri örneğini ayrı bir liste olarak al
    transactions = df["İçerik"].apply(lambda x: x.split(','))

    # TransactionEncoder kullanarak veriyi uygun formata dönüştür
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Frequent Pattern Growth (FP-Growth) algoritmasıyla sık kalıpları bul
    frequent_itemsets = fpgrowth(df_encoded, min_support=0.2, use_colnames=True)
    sorted_frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)

    # Destek değerini belirli bir ondalık hassasiyetle göster
    sorted_frequent_itemsets['support'] = sorted_frequent_itemsets['support'].apply(lambda x: round(x, 6))

    # Frozenset'i düz metin listesine dönüştür
    sorted_frequent_itemsets['itemsets'] = sorted_frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))

    # Frequent Pattern Growth (FP-Growth) algoritmasıyla elde edilen sonuçları CSV'ye yaz
    sorted_frequent_itemsets.to_csv(output_csv, index=False)

    # Başarıyla tamamlandı mesajı ver
    print("Frequent Patterns başarıyla CSV dosyasına yazıldı.")

if __name__ == "__main__":
    # Ana klasör ve çıkış klasör yollarını belirt
    root_folder = "news"  # Kullanılacak ana klasör yolunu güncelleyin

    # Çıkış klasörünü oluştur
    output_folder = create_output_folder()


    # Fonksiyonu çağır
    txt_to_csv(root_folder, output_folder)

    input_folder = output_folder  # Kullanılacak ana klasör yolunu güncelleyin
    output_folder1 = input_folder  # Çıkış klasör yolunu güncelleyin

    # Klasördeki tüm dosyaları al
    file_list = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    for file_name in file_list:
        input_file = os.path.join(input_folder, file_name)
        output_file = process_csv_file(input_file, output_folder1)

        # Çıktı dosyasının adını belirle
        output_csv = os.path.join(output_folder1, f"fp-growth_output_{file_name}")

        # Frequent Pattern Growth (FP-Growth) algoritmasıyla elde edilen sonuçları CSV'ye yaz
        apply_fp_growth(output_file, output_csv)

    print("İşlem tamamlandı.")
