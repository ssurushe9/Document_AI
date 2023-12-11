import os

def db_username():
    return os.environ['DB_USERNAME']

def db_password():
    return os.environ['DB_PASSWORD']

def db_endpoint():
    return os.environ['DB_ENDPOINT']

def db_port():
    return os.environ['DB_PORT']

def db_name():
    return os.environ['DB_NAME']