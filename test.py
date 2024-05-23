import spacy

# Загрузка моделей
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# Пример текста на немецком
text_de = "Angela Merkel ist die Bundeskanzlerin von Deutschland."
doc_de = spacy_de(text_de)

# Пример текста на английском
text_en = "Barack Obama was the 44th President of the United States."
doc_en = spacy_en(text_en)

# Вывод именованных сущностей в немецком тексте
for ent in doc_de.ents:
    print(ent.text, ent.label_)

# Вывод именованных сущностей в английском тексте
for ent in doc_en.ents:
    print(ent.text, ent.label_)
