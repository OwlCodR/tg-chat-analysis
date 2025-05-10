import ollama

HISTORY_FILE = 'chat_history.txt'
MODEL = 'deepseek-r1'

def run():
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        history = f.read().strip()

    prompt = f"""
    Ты мой друг. Отвечай так, как ответил бы он.
    Ответы должны быть очень похожи на реальные ответы из переписки, учти следующее:
    - Стиль общения, орфография, пунктуация и лексика
    - Используй ту же грамматику и пунктуацию
    - Сохраняй характерные слова и речевые обороты
    - Длину сообщений и их размер
    - Частота использования эмоджи

    Переписка:
    {history}

    Мы начинаем вести новую переписку, дальше отвечай без подписей:
    """

    while True:
        question = input('Запрос: ')
        prompt += f'\nМакс: {question}'

        print('Думаю...')
        response = ollama.generate(
            model=MODEL,
            prompt=prompt,
            options={'temperature': 0.6}
        )

        answer = response.get('response', '').strip()
        prompt += f'\nДаня: {answer}'

        print('Ответ: ', answer)

if __name__ == "__main__":
    run()