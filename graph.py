import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
import seaborn as sns

# CSV dosyasını oku
csv_file = 'output_4.csv'  # CSV dosyasının adını güncelleyin
df = pd.read_csv(csv_file)

# Her bir veri örneğini ayrı bir liste olarak al
transactions = df["İçerik"].apply(lambda x: x.split(','))

# TransactionEncoder kullanarak veriyi uygun formata dönüştür
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)


frequent_itemsets = fpgrowth(df_encoded, min_support=0.2, use_colnames=True)
sorted_frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=True)


# Destek değerini belirli bir ondalık hassasiyetle göster
sorted_frequent_itemsets['support'] = sorted_frequent_itemsets['support'].apply(lambda x: round(x, 6))

# Çıktıyı düzenle ve yazdır
output_df = sorted_frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
output_df = pd.DataFrame({'Support': sorted_frequent_itemsets['support'],'Frequent Patterns': output_df})
print(output_df)

# Frozenset'i düz metin listesine dönüştür
sorted_frequent_itemsets['itemsets'] = sorted_frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))


# Grafik için en çok kullanılan kelimeleri ve kelime çiftlerini seç
top_k_items = 21  # İlk 10 öğeyi seç
top_k_frequent_itemsets = sorted_frequent_itemsets.tail(top_k_items)

# Grafik oluştur
plt.figure(figsize=(12, 8))
sns.barplot(x='support', y='itemsets', data=top_k_frequent_itemsets, palette='viridis')
plt.xlabel('Support')
plt.ylabel('Frequent Itemsets')
plt.title(f'Top {top_k_items} Frequent Itemsets and Their Support')
plt.show()




