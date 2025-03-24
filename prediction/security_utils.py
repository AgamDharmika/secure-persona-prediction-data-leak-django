from Crypto.Cipher import AES
import base64
import os

SECRET_KEY = b"my_secret_encryption_key"  # Use a strong key (16, 24, or 32 bytes)

def pad(data):
    return data + (16 - len(data) % 16) * chr(16 - len(data) % 16)

def unpad(data):
    return data[:-ord(data[len(data) - 1:])]

def encrypt(data):
    cipher = AES.new(SECRET_KEY, AES.MODE_ECB)
    encrypted_data = cipher.encrypt(pad(data).encode())
    return base64.b64encode(encrypted_data).decode()

def decrypt(data):
    cipher = AES.new(SECRET_KEY, AES.MODE_ECB)
    decrypted_data = unpad(cipher.decrypt(base64.b64decode(data)).decode())
    return decrypted_data
