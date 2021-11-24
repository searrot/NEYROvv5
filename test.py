import asyncio
import aiohttp
from codetiming import Timer
import time
from urllib.parse import urlencode
import hmac, hashlib
import sqliter
import requests
import telebot
from config import IVAN_ID, RUSLAN_ID

admin_bot = telebot.TeleBot('1993230425:AAEqbDCNCDGDcAJ00w1nBmk9loenYbMRcbc')

async def task(name, work_queue, sym, price):
    print('salam')
    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
    async with aiohttp.ClientSession() as session:
        while not work_queue.empty():
            l = await work_queue.get()
            apikey = l[1]
            SECRET_KEY = l[2]
            timer.start()
            timestamp = int(time.time() * 1000)
            params1 = {
                'timestamp' : timestamp,    
            }
            query_string = urlencode(params1)
            params1['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
            headers = {
            'X-MBX-APIKEY': apikey
            }  
            async with session.get('https://api.binance.com/api/v3/account', headers=headers, params=params1) as response:
                args = await response.json()
                summa = float(args['balances'][11]['free'])
                print(summa, l[4])      
                params = {
                'symbol': sym+'USDT',
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': int(summa *l[4] / 100 / price),
                'timestamp': timestamp
                }
                query_string = urlencode(params)
                params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
                headers = {
                'X-MBX-APIKEY': apikey
                }                
                async with session.post('https://api.binance.com/api/v3/order', headers=headers, params=params) as response:
                    print(response.status)
                    print(response)
                    with open('log.txt', 'a') as file:
                        file.write(str(response))
                timer.stop()

async def task_sell(name, work_queue, sym):
    print('salam')
    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
    async with aiohttp.ClientSession() as session:
        while not work_queue.empty():
            l = await work_queue.get()
            apikey = l[1]
            SECRET_KEY = l[2]
            timer.start()
            timestamp = int(time.time() * 1000)
            params1 = {
                'timestamp' : timestamp,    
            }
            query_string = urlencode(params1)
            params1['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
            headers = {
            'X-MBX-APIKEY': apikey
            }  
            async with session.get('https://api.binance.com/api/v3/account', headers=headers, params=params1) as response:
                args = await response.json()
                balances = args['balances']
                for balance in balances:
                    if balance['asset'] == sym:
                        qua = float(balance['free'])
                params = {
                'symbol': sym+'USDT',
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': int(qua),
                'timestamp': timestamp
                }
                query_string = urlencode(params)
                params['signature'] = hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
                headers = {
                'X-MBX-APIKEY': apikey
                }                
                async with session.post('https://api.binance.com/api/v3/order', headers=headers, params=params) as response:
                    print(response.status)
                    with open('log.txt', 'a') as file:
                        file.write(str(response))
                timer.stop() 
 
async def main(sym, price):
    """
    Это основная точка входа в программу
    """

    # Создание очереди работы
    work_queue = asyncio.Queue()
    
    # Помещение работы в очередь
    for url in sqliter.all_info():
        await work_queue.put(url)
    
    # Запуск задач
    with Timer(text="\nTotal elapsed time: {:.1f}"):
        await asyncio.gather(
            asyncio.create_task(task("One", work_queue, sym, price)),
            asyncio.create_task(task("Two", work_queue, sym, price)),
            #asyncio.create_task(task("Three", work_queue, sym, price)),
            #asyncio.create_task(task("Four", work_queue, sym, price)),
            #asyncio.create_task(task("Five", work_queue, sym, price)),
            #asyncio.create_task(task("Six", work_queue, sym, price)),
            #asyncio.create_task(task("Seven", work_queue, sym, price)),
            #asyncio.create_task(task("Eight", work_queue, sym, price)),
            #asyncio.create_task(task("Nine", work_queue, sym, price))
            )


async def main1(sym):
    """
    Это основная точка входа в программу
    """
    # Создание очереди работы
    work_queue = asyncio.Queue()
 
    # Помещение работы в очередь
    for url in sqliter.all_info():
        await work_queue.put(url)
 
    # Запуск задач
    with Timer(text="\nTotal elapsed time: {:.1f}"):
        await asyncio.gather(
            asyncio.create_task(task_sell("One", work_queue, sym)),
            asyncio.create_task(task_sell("Two", work_queue, sym)),
            #asyncio.create_task(task_sell("Three", work_queue, sym)),
            #asyncio.create_task(task_sell("Four", work_queue, sym)),
            #asyncio.create_task(task_sell("Five", work_queue, sym)),
            #asyncio.create_task(task_sell("Six", work_queue, sym)),
            #asyncio.create_task(task_sell("Seven", work_queue, sym)),
            #asyncio.create_task(task_sell("Eight", work_queue, sym)),
            #asyncio.create_task(task_sell("Nine", work_queue, sym))
        ) 
 
def run(sym):
    req = requests.get('https://api.binance.com/api/v3/ticker/price?symbol='+sym+'USDT')
    price = float(req.json()['price'])
    admin_bot.send_message(IVAN_ID, 'Cryptobot: Установка покупки в список задач')
    admin_bot.send_message(RUSLAN_ID, 'Cryptobot: Установка покупки в список задач')
    asyncio.run(main(sym, price))
    admin_bot.send_message(IVAN_ID, 'Cryptobot: Совершена покупка ')
    admin_bot.send_message(RUSLAN_ID, 'Cryptobot: Совершена покупка')
    time.sleep(120)
    asyncio.run(main1(sym))
    admin_bot.send_message(IVAN_ID, 'Cryptobot: Совершена 1 продажа')
    admin_bot.send_message(RUSLAN_ID, 'Cryptobot: Совершена 1 продажа')
    time.sleep(10)
    asyncio.run(main1(sym))
    admin_bot.send_message(IVAN_ID, 'Cryptobot: Совершена 2 продажа')
    admin_bot.send_message(RUSLAN_ID, 'Cryptobot: Совершена 2 продажа')
    req = requests.get('https://api.binance.com/api/v3/ticker/price?symbol='+sym+'USDT')
    price1 = float(req.json()['price'])
    l = []
    l.append(price)
    l.append(price1)
    return l