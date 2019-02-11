def register(func):
    def wrapper(*args):
        print((func.__name__))
        return func(*args)
    return wrapper