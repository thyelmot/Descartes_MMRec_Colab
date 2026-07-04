import datetime

logmsg = ''
timemark = dict()
saveDefault = False
log_filepath = None  # Có thể set để lưu log ra file (VD: Google Drive path)

def set_log_file(filepath):
    """Set đường dẫn file log (dùng trên Colab để lưu vào Google Drive)."""
    global log_filepath
    log_filepath = filepath

def log(msg, save=None, oneline=False):
    global logmsg
    global saveDefault
    global log_filepath
    time = datetime.datetime.now()
    tem = '%s: %s' % (time, msg)
    if save != None:
        if save:
            logmsg += tem + '\n'
    elif saveDefault:
        logmsg += tem + '\n'
    if oneline:
        print(tem, end='\r')
    else:
        print(tem)
    # Ghi ra file nếu có set filepath
    if log_filepath and not oneline:
        try:
            with open(log_filepath, 'a', encoding='utf-8') as f:
                f.write(tem + '\n')
        except Exception:
            pass

def marktime(marker):
    global timemark
    timemark[marker] = datetime.datetime.now()


if __name__ == '__main__':
    log('')
