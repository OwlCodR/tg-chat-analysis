import re
from collections import Counter
import pymorphy3
import ollama
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import os

# ======================
# КОНФИГУРАЦИОННЫЕ НАСТРОЙКИ
# ======================

# Идентификаторы участников
USER_ALIAS = 'Я'              # Ваше имя в переписке
FRIEND_ALIAS = 'Друг'         # Имя друга в переписке

FILE_TAG = 'friend'

# Настройки файлов
INPUT_FILE = f'{FILE_TAG}_chat_history.txt'             # Файл с перепиской
OUTPUT_REPORT = f'{FILE_TAG}_chat_analysis_report.txt'  # Файл отчета
WORDCLOUD_FILE = f'{FILE_TAG}_wordcloud.png'            # Облако слов
LENGTH_PLOT_FILE = f'{FILE_TAG}_message_lengths.png'    # График длины сообщений
POS_PLOT_FILE = f'{FILE_TAG}_pos_distribution.png'      # Распределение частей речи

# Настройки анализа
OLLAMA_MODEL = 'deepseek-r1'
MIN_WORD_LENGTH = 1           # Минимальная длина слова
TOP_WORDS_LIMIT = 10          # Количество топовых слов
MESSAGE_SAMPLE_SIZE = 500     # Сообщений для эмоционального анализа

# Фильтрация частей речи (pymorphy3 tags)
FILTER_POS_TAGS = {
    'PREP',  # Предлоги
    'CONJ',  # Союзы
    'PRCL',  # Частицы
    'INTJ',  # Междометия
    'NPRO',  # Местоимения
    'NUMR',  # Числительные
    'PRED',  # Предикативы
}

# Пользовательские стоп-слова
CUSTOM_STOPWORDS = {
    'это', 'вот', 'ну', 'да', 'нет', 'не', 'ли', 'же', 'бы', 'то',
    'как', 'так', 'уже', 'еще', 'или', 'но', 'за', 'из', 'от', 'до',
    'и', 'у', 'на', 'ты', 'я', 'что', 'там', 'если', 'в', 'все',
    'а', 'по', 'с', 'мне', 'если', 'есть', 'для', 'какой',
    'такой', 'весь', 'который', 'быть', 'http', 'ru', 'com', 'pikabu',
    'utm_medium', '>youtu', 'story', 't', 'https', 'utm_source', 'этот',
    'тут', 'www', 'об', 'тот'
}

# Настройки визуализации
PLOT_COLORS = {
    'user': '#4e79a7',
    'friend': '#e15759',
    'other': '#59a14f'
}
FONT_PATH = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
plt.style.use('ggplot')

# ======================
# КЛАСС АНАЛИЗА ПЕРЕПИСКИ
# ======================
class ChatAnalyzer:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.messages = []
        self.stopwords = self._init_stopwords()
        self.pos_mapping = {
            'NOUN': 'Существительные',
            'VERB': 'Глаголы',
            'ADJF': 'Прилагательные',
            'ADVB': 'Наречия',
            'INFN': 'Инфинитивы',
            'GRND': 'Деепричастия',
            'PRTF': 'Причастия'
        }

    def _init_stopwords(self):
        """Инициализация стоп-слов с учетом всех словоформ"""
        base_words = set(CUSTOM_STOPWORDS)
        
        for word in CUSTOM_STOPWORDS:
            try:
                parsed = self.morph.parse(word)[0]
                base_words.update([f.word for f in parsed.lexeme])
            except:
                continue
                
        return base_words

    def _filter_word(self, word):
        """Комплексная фильтрация слова"""
        if len(word) < MIN_WORD_LENGTH:
            return False
            
        parsed = self.morph.parse(word)[0]
        pos = str(parsed.tag).split(',')[0]
        
        if pos in FILTER_POS_TAGS:
            return False
            
        if parsed.normal_form in self.stopwords:
            return False
            
        return True

    def load_chat(self):
        """Загрузка и парсинг переписки с фильтрацией"""
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"Файл переписки {INPUT_FILE} не найден")
        
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:                    
                if match := re.match(rf'^({USER_ALIAS}|{FRIEND_ALIAS}):\s*(.+)', line.strip()):
                    self.messages.append({
                        'sender': match.group(1),
                        'text': match.group(2),
                        'words': [ word if word != 'мочь' else 'можно' for word in self._process_text(match.group(2)) ]
                    })

    def _process_text(self, text):
        """Обработка текста с морфологическим анализом"""
        words = re.findall(rf'\b[а-яё]{{{MIN_WORD_LENGTH},}}\b', text.lower())
        return [
            self.morph.parse(w)[0].normal_form 
            for w in words 
            if self._filter_word(w)
        ]

    def basic_stats(self):
        """Базовая статистика переписки"""
        stats = {
            'total': len(self.messages),
            'user': sum(1 for m in self.messages if m['sender'] == USER_ALIAS),
            'message_lengths': {
                'user': [len(m['text']) for m in self.messages if m['sender'] == USER_ALIAS],
                'friend': [len(m['text']) for m in self.messages if m['sender'] == FRIEND_ALIAS]
            },
            'word_counts': {
                'user': sum(len(m['words']) for m in self.messages if m['sender'] == USER_ALIAS),
                'friend': sum(len(m['words']) for m in self.messages if m['sender'] == FRIEND_ALIAS)
            }
        }
        
        stats.update({
            'user_ratio': stats['user'] / stats['total'],
            'avg_length': {
                'user': np.mean(stats['message_lengths']['user']) if stats['message_lengths']['user'] else 0,
                'friend': np.mean(stats['message_lengths']['friend']) if stats['message_lengths']['friend'] else 0
            },
            'avg_words': {
                'user': stats['word_counts']['user'] / stats['user'] if stats['user'] else 0,
                'friend': stats['word_counts']['friend'] / (stats['total'] - stats['user']) if stats['total'] != stats['user'] else 0
            }
        })
        
        return stats

    def vocabulary_analysis(self):
        """Комплексный анализ словарного запаса"""
        def analyze(words):
            cnt = Counter(words)
            unique = set(words)
            
            # Статистика по частям речи
            pos_stats = Counter()
            for word in words:
                parsed = self.morph.parse(word)[0]
                main_pos = str(parsed.tag).split(',')[0]
                pos_stats[main_pos] += 1
            
            return {
                'total': len(words),
                'unique': len(unique),
                'diversity': len(unique) / len(words) if words else 0,
                'top': cnt.most_common(TOP_WORDS_LIMIT),
                'pos_stats': pos_stats.most_common(10)
            }

        words = {
            'user': [w for m in self.messages if m['sender'] == USER_ALIAS for w in m['words']],
            'friend': [w for m in self.messages if m['sender'] == FRIEND_ALIAS for w in m['words']],
            'all': [w for m in self.messages for w in m['words']]
        }

        return {
            'user': analyze(words['user']),
            'friend': analyze(words['friend']),
            'all': analyze(words['all'])
        }

    def emotional_analysis(self):
        """Анализ эмоционального тона через LLM"""
        sample = '\n'.join(
            f"{m['sender']}: {m['text']}" 
            for m in self.messages[-MESSAGE_SAMPLE_SIZE:]
        )
        
        prompt = f"""Проанализируй эмоциональный тон в этом диалоге:
{sample}

Ответь в формате:
- Основные эмоции: 
- Ключевые слова эмоций: 
- Общий тон (позитивный/нейтральный/негативный): 
- Интенсивность (1-10): """
        
        try:
            response = ollama.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                options={'temperature': 0.4}
            )
            return response['response']
        except Exception as e:
            return f"⚠ Ошибка анализа эмоций: {str(e)}"

    def generate_wordcloud(self):
        """Генерация облака слов (исправленная версия)"""
        text = ' '.join(' '.join(m['words']) for m in self.messages)
        
        wordcloud = WordCloud(
            width=2560,
            height=1440,
            background_color='white',
            colormap='viridis',
            font_path=FONT_PATH,
            stopwords=self.stopwords,
            max_words=300,
            collocations=False
        ).generate(text)
        
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(WORDCLOUD_FILE, bbox_inches='tight')
        plt.close()

    def plot_message_lengths(self):
        """Визуализация распределения длины сообщений (оптимизированная)"""
        stats = self.basic_stats()
        
        plt.figure(figsize=(12, 6))
        bins = np.linspace(0, 200, 20)
        
        # Оптимизированное построение гистограммы
        plt.hist(
            [stats['message_lengths']['user'], stats['message_lengths']['friend']],
            bins=bins,
            label=[USER_ALIAS, FRIEND_ALIAS],
            color=[PLOT_COLORS['user'], PLOT_COLORS['friend']],
            alpha=0.6
        )
        
        plt.xlabel('Длина сообщения (символов)', fontsize=12)
        plt.ylabel('Количество сообщений', fontsize=12)
        plt.title('Распределение длины сообщений', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Исправленное сохранение
        plt.savefig(LENGTH_PLOT_FILE, bbox_inches='tight')
        plt.close()

    def plot_pos_distribution(self):
        """Визуализация распределения частей речи"""
        vocab = self.vocabulary_analysis()
        
        # Подготовка данных
        pos_data = {
            'user': dict(vocab['user']['pos_stats']),
            'friend': dict(vocab['friend']['pos_stats'])
        }
        
        # Собираем все части речи
        all_pos = set(pos_data['user'].keys()).union(set(pos_data['friend'].keys()))
        
        # Преобразуем в читаемые названия
        readable_pos = []
        for pos in all_pos:
            readable_pos.append(self.pos_mapping.get(pos, pos))
        
        # Значения для каждого участника
        user_counts = [pos_data['user'].get(pos, 0) for pos in all_pos]
        friend_counts = [pos_data['friend'].get(pos, 0) for pos in all_pos]
        
        # Построение графика
        x = np.arange(len(readable_pos))
        width = 0.35
        
        plt.figure(figsize=(14, 7))
        plt.bar(x - width/2, user_counts, width, label=USER_ALIAS, color=PLOT_COLORS['user'])
        plt.bar(x + width/2, friend_counts, width, label=FRIEND_ALIAS, color=PLOT_COLORS['friend'])
        
        plt.xlabel('Часть речи', fontsize=12)
        plt.ylabel('Количество', fontsize=12)
        plt.title('Распределение частей речи', fontsize=14)
        plt.xticks(x, readable_pos, rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(POS_PLOT_FILE, bbox_inches='tight', dpi=150)
        plt.close()

    def generate_report(self):
        """Генерация комплексного отчета"""
        print('Получение статистики...')
        stats = self.basic_stats()

        print('Анализ словарного запаса...')
        vocab = self.vocabulary_analysis()

        print('Эмоциональный анализ...')
        emotion = self.emotional_analysis()
        
        # Формирование отчета
        report = [
            "="*60,
            f"📊 КОМПЛЕКСНЫЙ АНАЛИЗ ПЕРЕПИСКИ".center(60),
            f"({datetime.now().strftime('%d.%m.%Y %H:%M')})".center(60),
            "="*60,
            "\n📌 ОСНОВНЫЕ ПОКАЗАТЕЛИ",
            "-"*40,
            f"▪ Всего сообщений: {stats['total']}",
            f"▪ Сообщений {USER_ALIAS}: {stats['user']} ({stats['user_ratio']:.1%})",
            f"▪ Сообщений {FRIEND_ALIAS}: {stats['total'] - stats['user']} ({(1 - stats['user_ratio']):.1%})",
            f"\n▪ Средняя длина сообщения:",
            f"  ▸ {USER_ALIAS}: {stats['avg_length']['user']:.1f} символов",
            f"  ▸ {FRIEND_ALIAS}: {stats['avg_length']['friend']:.1f} символов",
            f"\n▪ Среднее количество слов на сообщение:",
            f"  ▸ {USER_ALIAS}: {stats['avg_words']['user']:.1f} слов",
            f"  ▸ {FRIEND_ALIAS}: {stats['avg_words']['friend']:.1f} слов",
            
            "\n\n📚 ЛЕКСИЧЕСКИЙ АНАЛИЗ",
            "-"*40,
            f"▪ Всего слов (без стоп-слов): {vocab['all']['total']}",
            f"▪ Уникальных слов: {vocab['all']['unique']}",
            f"▪ Лексическое разнообразие (TTR): {vocab['all']['diversity']:.1%}",
            f"\n▪ Словарный запас:",
            f"  ▸ {USER_ALIAS}: {vocab['user']['unique']} уникальных слов",
            f"  ▸ {FRIEND_ALIAS}: {vocab['friend']['unique']} уникальных слов",
            
            f"\n\n🏆 ТОП-{TOP_WORDS_LIMIT} СЛОВ",
            "-"*40,
            f"▪ {USER_ALIAS}:",
            "  " + ", ".join(f"{w}({c})" for w, c in vocab['user']['top']),
            f"\n▪ {FRIEND_ALIAS}:",
            "  " + ", ".join(f"{w}({c})" for w, c in vocab['friend']['top']),
                        
            "\n\n🎭 ЭМОЦИОНАЛЬНЫЙ АНАЛИЗ",
            "-"*40,
            emotion,
            
            "\n" + "="*60
        ]
        
        # Сохранение отчета
        with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        
        # Генерация графиков
        self.generate_wordcloud()
        self.plot_message_lengths()
        self.plot_pos_distribution()

# ======================
# ЗАПУСК АНАЛИЗА
# ======================
if __name__ == "__main__":
    print("="*60)
    print("🔍 АНАЛИЗ ПЕРЕПИСКИ".center(60))
    print("="*60)
    
    try:
        analyzer = ChatAnalyzer()
        
        print("\n📥 Загрузка данных...")
        analyzer.load_chat()
        
        print("📊 Анализ статистики...")
        stats = analyzer.basic_stats()
        print(f"▪ Обработано сообщений: {stats['total']}")
        print(f"▪ Соотношение {USER_ALIAS}/{FRIEND_ALIAS}: {stats['user_ratio']:.1%}")
        
        print("\n📚 Анализ словарного запаса...")
        vocab = analyzer.vocabulary_analysis()
        print(f"▪ Уникальных слов: {vocab['all']['unique']}")
        print(f"▪ Лексическое разнообразие: {vocab['all']['diversity']:.1%}")
        
        print("\n🎭 Анализ эмоционального тона...")
        print(analyzer.emotional_analysis().split('\n')[0])
        
        print("\n🖼️ Генерация визуализаций...")
        analyzer.generate_report()
        
        print("\n✅ АНАЛИЗ ЗАВЕРШЕН".center(60))
        print("="*60)
        print(f"\nСозданы файлы:")
        print(f"- Текстовый отчет: {OUTPUT_REPORT}")
        print(f"- Облако слов: {WORDCLOUD_FILE}")
        print(f"- График длины сообщений: {LENGTH_PLOT_FILE}")
        print(f"- Распределение частей речи: {POS_PLOT_FILE}")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {str(e)}")
        print("Проверьте наличие файла переписки и настройки")