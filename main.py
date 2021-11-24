import socket, requests
import test
import sqliter
from settings import bot 
import telebot
from config import IVAN_ID, RUSLAN_ID
import logging

logging.basicConfig(level=logging.DEBUG)

admin_bot = telebot.TeleBot('1993230425:AAEqbDCNCDGDcAJ00w1nBmk9loenYbMRcbc')

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 2000))
    s.listen(4)
    conn, addr = s.accept()
except Exception as e:
    print(e)
    admin_bot.send_message(IVAN_ID, f'Cryptobot: ошибка сокета: {e}')
    admin_bot.send_message(RUSLAN_ID, f'Cryptobot: ошибка сокета: {e}')

while True:
    try:
        try:
            data = conn.recv(1024).decode('latin-1')
        except:
            data = conn.recv(1024).decode('utf-8')
        if not data:
            admin_bot.send_message(RUSLAN_ID, 'Cryptobot:Ошибка получения запроса от сервера, пустой запрос')
            admin_bot.send_message(IVAN_ID, 'Cryptobot:Ошибка получения запроса от сервера, пустой запрос')
            raise SystemExit
        HDRS = 'HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n'
        content = 'Its okay'.encode('utf-8')
        conn.send(HDRS.encode('utf-8') + content)
        if data.split(' ')[1].split('/')[1] == 'ZldaOUMyTlBiU1hFdWpYRkZUbUFFNjdv':
            sym = data.split(' ')[1].split('/')[2]
            print(f'GET-запрос к серверу: {sym}')

            if sym == 'DOGE' or sym == "SHIB":
                try:
                    pr = test.run(data.split(' ')[1].split('/')[2])
                except Exception as e:
                    s.close()
                    admin_bot.send_message(IVAN_ID, f'Cryptobot:Ошибка в скрипте покупки: {e}')
                    admin_bot.send_message(RUSLAN_ID, f'Cryptobot:Ошибка в скрипте покупки: {e}')
                    conn.close()
                    s.close()
                    raise SystemExit

                price = '{:0.9f}'.format(pr[0])
                price1 ='{:0.9f}'.format(pr[1])
                procent = '{:0.2}'.format(float(price1)/float(price)*100-100)
                print(price, price1, procent)
                l = sqliter.ids()# русским
                l1 = sqliter.ids_en()
                for i in l:
                    bot.send_message(chat_id=i, text = 'Была совершена покупка ' + sym + ' по цене ' + str(price) + ' и продажа по цене ' + str(price1) + '\n✅ Прибыль ' + procent + ' процента')
                for n in l1:
                    bot.send_message(chat_id=n, text = 'A purchase was made ' + sym + ' at the price ' + str(price) + ' and selling at a price ' + str(price1) + '\n✅ Profit ' + procent + ' procent')
        
            elif sym == "TEST":
                admin_bot.send_message(IVAN_ID, 'Cryptobot:Сервер сryptobot находится в рабочем состоянии')
                admin_bot.send_message(RUSLAN_ID, 'Cryptobot:Сервер сryptobot находится в рабочем состоянии')

        conn.close()
        conn, addr = s.accept()

    except Exception as e:
        admin_bot.send_message(IVAN_ID, f'Cryptobot:Посторонняя ошибка: {e}')
        admin_bot.send_message(RUSLAN_ID, f'Cryptobot:Посторонняя ошибка: {e}')
        raise SystemExit
