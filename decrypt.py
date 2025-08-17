import argparse
import hmac
import hashlib
import os
from typing import Union, List
# from Cryptodome.Cipher import AES

from Crypto.Cipher import AES # 如果上面的导入失败，可以尝试使用这个

SQLITE_FILE_HEADER = "SQLite format 3\x00"  # SQLite文件头

KEY_SIZE = 32
DEFAULT_PAGESIZE = 4096
DEFAULT_ITER = 64000


# 通过密钥解密数据库
def decrypt(key: str, db_path, out_path):
    """
    通过密钥解密数据库
    :param key: 密钥 64位16进制字符串
    :param db_path:  待解密的数据库路径(必须是文件)
    :param out_path:  解密后的数据库输出路径(必须是文件)
    :return:
    """
    if not os.path.exists(db_path) or not os.path.isfile(db_path):
        return False, f"[-] db_path:'{db_path}' File not found!"
    if not os.path.exists(os.path.dirname(out_path)):
        return False, f"[-] out_path:'{out_path}' File not found!"

    if len(key) != 64:
        return False, f"[-] key:'{key}' Len Error!"

    password = bytes.fromhex(key.strip())
    with open(db_path, "rb") as file:
        blist = file.read()

    salt = blist[:16]
    byteKey = hashlib.pbkdf2_hmac("sha1", password, salt, DEFAULT_ITER, KEY_SIZE)
    first = blist[16:DEFAULT_PAGESIZE]
    if len(salt) != 16:
        return False, f"[-] db_path:'{db_path}' File Error!"

    mac_salt = bytes([(salt[i] ^ 58) for i in range(16)])
    mac_key = hashlib.pbkdf2_hmac("sha1", byteKey, mac_salt, 2, KEY_SIZE)
    hash_mac = hmac.new(mac_key, first[:-32], hashlib.sha1)
    hash_mac.update(b'\x01\x00\x00\x00')

    if hash_mac.digest() != first[-32:-12]:
        return False, f"[-] Key Error! (key:'{key}'; db_path:'{db_path}'; out_path:'{out_path}' )"

    newblist = [blist[i:i + DEFAULT_PAGESIZE] for i in range(DEFAULT_PAGESIZE, len(blist), DEFAULT_PAGESIZE)]

    with open(out_path, "wb") as deFile:
        deFile.write(SQLITE_FILE_HEADER.encode())
        t = AES.new(byteKey, AES.MODE_CBC, first[-48:-32])
        decrypted = t.decrypt(first[:-48])
        deFile.write(decrypted)
        deFile.write(first[-48:])

        for i in newblist:
            t = AES.new(byteKey, AES.MODE_CBC, i[-48:-32])
            decrypted = t.decrypt(i[:-48])
            deFile.write(decrypted)
            deFile.write(i[-48:])
    return True, [db_path, out_path, key]
print(decrypt('77ce67a8a36c47ebb032ff6fd04300a7d16xxxxxxxxxxxxxxxxxxxxxxxxxx', ".\MSG0.db", ".\\decrypt_MSG0.db"))