from telethon.sync import TelegramClient

# --- Конфиг ---
API_ID = 0                          # Заменить на свой API_ID
API_HASH = ''                       # Подставить свой API_HASH
TARGET_USER = ''                    # @tag или имя в контактах

LIMIT = 100000
BLACKLIST = ['busy']
OUTPUT_FILE = 'chat_history.txt'
USER_ALIAS = 'Я'
FRIEND_ALIAS = 'Друг'

with TelegramClient('session', API_ID, API_HASH) as client:
    target = client.get_entity(TARGET_USER)
    
    print('Получаю сообщения...')
    messages = list(client.iter_messages(target, limit=LIMIT))

    print('Делаю реверс...')
    messages.reverse()

    print('Сохраняю в файл...')
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:        
        for msg in messages:
            sender = USER_ALIAS if msg.out else FRIEND_ALIAS

            if msg.text is None or len(msg.text) == 0:
                continue
            
            hasBlacklistedWord = False
            for word in BLACKLIST:
                if word in msg.text:
                    hasBlacklistedWord = True
                    break
            
            if hasBlacklistedWord:
                continue

            f.write(f"{sender}: {msg.text}\n")

print(f"✅ Переписка сохранена в {OUTPUT_FILE}")
