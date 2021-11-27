#!/usr/bin/env python3

# Copyright (C) 2021 megumifox <i@megumifox.com>

from telethon import TelegramClient, events, sync
from telethon.tl.types import PeerUser, PeerChat, PeerChannel
from telethon.tl.functions.channels import EditPhotoRequest
from telethon.tl.types import InputChatUploadedPhoto
from datetime import datetime, timedelta, timezone
import logging
import os
import cv2
import numpy as np
import asyncio
import re
import requests 
import magic
import yaml
from linkpreview import link_preview
import random

headers = {'User-Agent': 'Twitterbot'}
artworks_root = ""
last_all = []

logging.basicConfig(format='[%(levelname) 5s/%(asctime)s] %(name)s: %(message)s',
                    level=logging.INFO)
logger=logging.getLogger("AVATAR_MAIN")

# load config
with open('config.yaml', 'r') as f:
    bot_config = yaml.load(f,Loader=yaml.SafeLoader)

client = TelegramClient(bot_config['bot_name'],
                        bot_config['app_id'],
                        bot_config['api_hash'],
                        )
client.start(bot_token=bot_config['bot_token'])

# bot config done

def is_chat_in_whitelist(chat_id):
    for channel in bot_config['chat_settings']:
        if channel['id'] == chat_id:
            return True
    return False

def get_chat_time_limit(chat_id):
    for channel in bot_config['chat_settings']:
        if channel['id'] == chat_id:
            return channel['time_limit']
    return None

async def get_args(event):
    rgb = (255,255,255)
    if '#' in event.raw_text:
        try:
            hexrgb =str(event.raw_text).split("#")[1].split(' ')[0]
            logger.debug(hexrgb)
            rgb = tuple(int(hexrgb[i:i+2], 16) for i in (0, 2, 4))
        except Exception as e:
            logger.error(e)
            m = await event.reply("參數格式錯誤！")
    print(rgb)
    return rgb

async def is_timeup(event):
    global last_all
    for chat in last_all:
        if chat['chat_id'] == event.chat_id:
            last = chat['last']
            chat_time_limit = get_chat_time_limit(event.chat_id)
            if  chat['processing']:
                m = await event.reply("´_>`,在跑了，在跑了（")
                return False
            if (event.message.date - last) < timedelta(seconds = chat_time_limit):
                t = (timedelta(seconds = chat_time_limit) - (event.message.date -last)).total_seconds()
                m = await event.reply("賢者時間還剩"+str(t)+"s")
                return False
            else:
                chat['processing'] = True
                return True
    last = datetime.now(timezone.utc) - timedelta(hours = 2)
    dictadd = {'chat_id': event.chat_id, 'last': last,'processing':True}
    last_all.append(dictadd)
    return True

def update_last(event,processing_only=False):
    global last_all
    for chat in last_all:
        if chat['chat_id'] == event.chat_id:
            chat['processing'] = False
            if not processing_only:
                chat['last'] = event.message.date
    return 0

def video2img(file):
    try:
        videoCapture = cv2.VideoCapture(file)
        length = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        seekframe = random.randint(0,length-1)
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES,seekframe)
        success, frame = videoCapture.read()
        image = frame
    except Exception as e:
        logger.error(e)
        image = -1
    finally:
        videoCapture.release()
        os.remove(file)
        return image

def img_resize(img):
    rows,cols,channels = img.shape
    if rows > cols:
        scale_factor = 2048/cols
    else:
        scale_factor = 2048/rows
    img = cv2.resize(img,None,fx=scale_factor,fy=scale_factor,interpolation=cv2.INTER_AREA)
    return img
    
def img_animeface_detect(image,cascade_file = "./lbpcascade_animeface.xml"):
    rows,cols,channels = image.shape
    logger.debug(image.shape)
    cascade = cv2.CascadeClassifier(cascade_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    faces_num = len(faces)
    if faces_num != 0:
        logger.info("%s animeface detected!",faces_num)
        face_x = 0
        face_y = 0
        logger.debug(faces)
        for face in faces:
            x, y, w, h = face
            face_x += (x+w/2)/faces_num
            face_y += (y+h/2)/faces_num
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        face_x = int(face_x)
        face_y = int(face_y)
        if rows > cols:
            cxmin = 0
            cxmax = cols
            cymin = (face_y - int(cols/2))
            cymax = (face_y + int(cols/2))
            if cymin < 0:
                cymin = 0
                cymax = cols
            if cymax > rows:
                cymin = rows - cols
                cymax = rows
        else:
            cxmin = (face_x - int(rows/2))
            cxmax = (face_x + int(rows/2))
            cymin = 0
            cymax = rows
            if cxmin < 0:
                cxmin = 0
                cxmax = rows
            if cxmax > cols:
                cxmin = cols - rows
                cxmax = cols
        logger.debug((cxmin,cxmax,cymin,cymax))
        img = image[cymin:cymax,cxmin:cxmax,:]
        return img
    else:
        logger.info("no animeface detected!")
        return image



async def add_img_bcakground(event,img,rgb):
    r,g,b = rgb
    rows,cols,channels = img.shape
    if channels != 4:
        return img

    result = np.zeros((rows, cols, 3), np.uint8)
    alpha = img[:, :, 3] / 255.0
    result[:, :, 0] = (1. - alpha) * b + alpha * img[:, :, 0]
    result[:, :, 1] = (1. - alpha) * g + alpha * img[:, :, 1]
    result[:, :, 2] = (1. - alpha) * r + alpha * img[:, :, 2]
    return result

async def get_link_image(event):
    replymsg = await event.message.get_reply_message()
    try:
        link_ids = re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+',replymsg.message)
        logger.debug(link_ids)
    except Exception as e:
        return 1
    if link_ids == []:
        return 1
    mr = await event.reply("Downloading ...")
    try:
        r = requests.get(link_ids[0],headers=headers) 
        f_mime_type = magic.detect_from_content(r.content).mime_type
        if "html" in f_mime_type:
            preview = link_preview("http://localhost", r.text,parser="lxml")
            r = requests.get(preview.image,headers=headers)
            f_mime_type = magic.detect_from_content(r.content).mime_type
        if "image" in f_mime_type:
            logger.debug(f_mime_type)
            img_buf = np.asarray(bytearray(r.content), dtype="uint8")
            image = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
        else:
            raise Exception('UNKNOW!')
    except BaseException as e:
        logger.error(e)
        m = await event.reply("´_>`, {}".format(e))
        return -1
    finally:
        await mr.delete()
    return image

async def get_pixiv_image(event):
    pic_num = 0
    if '!' in event.raw_text:
        try:
            pic_num = int(str(event.raw_text).split("!")[1].split(' ')[0])
        except Exception as e:
            m = await event.reply("參數格式錯誤！")
            pic_num = 0
    replymsg = await event.message.get_reply_message()
    try:
        image_ids = re.findall('(\d{6,})',replymsg.message)
        logger.debug(image_ids)
    except Exception as e:
        m = await event.reply("你怎麼什麼都沒給")
        return 1
    if image_ids == []:
        m = await event.reply("找不到圖片id")
        return 1
    mr = await event.reply("Downloading ...")
    try:
        if pic_num == 0:
            r = requests.get(artworks_root+image_ids[0]+'.png')
            f_mime_type = magic.detect_from_content(r.content).mime_type
        else:
            f_mime_type = "html"
        if "html"  in f_mime_type:
            r = requests.get(artworks_root+image_ids[0]+'-{}.png'.format(pic_num+1),headers=headers)
            f_mime_type = magic.detect_from_content(r.content).mime_type
        if "image" in f_mime_type:
            logger.debug(f_mime_type)
            img_buf = np.asarray(bytearray(r.content), dtype="uint8")
            image = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
        else:
            raise Exception('UNKNOW!')
    except BaseException as e:
        logger.error(e)
        m = await event.reply("´_>`, {}".format(e))
        return -1
    finally:
        await mr.delete()
    return image

async def get_telegram_img(event):
        replymsg = await event.message.get_reply_message()
        logger.debug(replymsg.file.mime_type)
        logger.debug(replymsg.file.name)
        if replymsg.file.size > 15*1024**2:
            m = await event.reply("不要啊啊啊啊，太大了！！！")
            return 0
        try:
            if (replymsg.file.mime_type == "image/gif") or ("video" in replymsg.file.mime_type):
                file_name = '/tmp/video{}.tmp'.format(event.chat_id)
                await client.download_media(message=replymsg,file=file_name)
                image = video2img(file_name)
                return image
            if 'image' in replymsg.file.mime_type:
                file_name = '/tmp/image{}.tmp'.format(event.chat_id)
                buf_bytes = await client.download_media(message=replymsg,file=bytes)
                img_buf = np.asarray(bytearray(buf_bytes), dtype="uint8")
                image = cv2.imdecode(img_buf, cv2.IMREAD_UNCHANGED)
                return image
            else:
                m = await event.reply("不支持的類型,{}".format(replymsg.file.mime_type))
                return -1
        except Exception as e:
            logger.error(e)
            m = await event.reply("出錯了！".format(e))
            return -1

async def get_img(event,auto_detect):
    if auto_detect == 'pixiv':
        img = await get_pixiv_image(event)
    if auto_detect == 'link':
        img = await get_link_image(event)
    if auto_detect == None:
        replymsg = await event.message.get_reply_message()
        if replymsg is None:
            img = await get_link_image(event)
        elif replymsg.file is None:
            img = await get_link_image(event)
        else:
            img = await get_telegram_img(event)
    return img
    
async def avatar(event,
                 animeface_detect=False,
                 auto_detect=None):
    sender = await event.get_sender()
    logger.info("sender_id = %s,username= %s,sender first_name = %s,last_name=%s, message = %s,chat_id= %s",
                event.message.from_id,
                sender.username,sender.first_name,
                sender.last_name,
                event.message.message,
                event.chat_id)
    # check chat in chat whitelist
    if not is_chat_in_whitelist(event.chat_id):
        m = await event.reply("如果需要使用請先聯系 bot 管理員將該羣加入白名單。chat_id={}".format(event.chat_id))
        return -1
    # check time limit 
    if not await is_timeup(event):
        return -1

    #waitmsg = await event.reply("處理中...")
    try:
        # get img
        img = await get_img(event,auto_detect)

        if isinstance(img,int):
            if img == 1:
                m = await event.reply("你的頭呢？")
            raise Exception('Can not get image!')

        rgb = await get_args(event)

        # img channl = 4 add background
        img = await add_img_bcakground(event,img,rgb)

        # resize image
        img = img_resize(img)
        
        # img animeface_detect
        if animeface_detect:
            img = img_animeface_detect(img)
        
        file_name = '/tmp/avatar{}.jpg'.format(event.chat_id)
        cv2.imwrite(file_name,img)

        upload_file_result = await client.upload_file(file_name)
        os.remove(file_name)
        input_chat_uploaded_photo = InputChatUploadedPhoto(upload_file_result)
        result = await client(EditPhotoRequest(channel=event.message.to_id,
        photo=input_chat_uploaded_photo))
        update_last(event)
        logger.info("success,chat_id = %s",event.chat_id)
    except Exception as e:
        logger.error(e)
    finally:
        update_last(event,True)
        #await waitmsg.delete()
    return 0

# Telegram commands handler

@client.on(events.NewMessage(pattern=r'/start'))
async def handler(event):
    if '@' in event.message.message and not event.message.mentioned and bot_config['bot_name'] not in event.message.message:
        return -1
    m = await event.reply("如果需要使用請先聯系 bot 管理員，如果已經在使用請忽略本消息。chat_id={}".format(event.chat_id))


@client.on(events.NewMessage(pattern=r'/info'))
async def handler(event):
    if '@' in event.message.message and not event.message.mentioned and bot_config['bot_name'] not in event.message.message:
        return -1
    m = await event.reply(u"本機器人提供羣組換頭功能，支持視頻，GIF(隨機幀),表情和正常的圖片(不支持動態表情)。\n 用法爲引用需要更換的消息（消息中包含圖片視頻等）然後使用 avatar 命令\n 命令後可帶參數 #RRGGBB 用於設定透明圖片的背景顏色。\n Source code: https://github.com/MarvelousBlack/avatar2")

@client.on(events.NewMessage(pattern=r'/chat_id'))
async def handler(event):
    if '@' in event.message.message and not event.message.mentioned and bot_config['bot_name'] not in event.message.message:
        return -1
    m = await event.reply("chat_id={}".format(event.chat_id))

@client.on(events.NewMessage(func=lambda e: not e.is_private,pattern=r'/avatar_white'))
async def handler(event):
    if '@' in event.message.message and not event.message.mentioned and bot_config['bot_name'] not in event.message.message:
        return -1
    await avatar(event)

@client.on(events.NewMessage(func=lambda e: not e.is_private,pattern=r'/link'))
async def handler(event):
    if '@' in event.message.message and not event.message.mentioned and bot_config['bot_name'] not in event.message.message:
        return -1
    await avatar(event,auto_detect="link")

@client.on(events.NewMessage(func=lambda e: not e.is_private,pattern=r'/pixiv_id'))
async def handler(event):
    if '@' in event.message.message and not event.message.mentioned and bot_config['bot_name'] not in event.message.message:
        return -1
    await avatar(event,animeface_detect=True,auto_detect="pixiv")

@client.on(events.NewMessage(func=lambda e: not e.is_private,pattern=r'/avatar_head_detect'))
async def handler(event):
    if '@' in event.message.message and not event.message.mentioned and bot_config['bot_name'] not in event.message.message:
        return -1
    await avatar(event,animeface_detect=True,auto_detect=None)

if __name__ == "__main__":
    try:
        client.run_until_disconnected()
    finally:
        client.disconnect()

