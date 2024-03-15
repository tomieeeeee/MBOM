import uuid
def get_mac_address_error():
    '''   error function
    uuid.getnode return a random number when don't find node
    '''
    mac=uuid.UUID(int = uuid.getnode()).hex[-12:] 
    return ":".join([mac[e:e+2] for e in range(0,11,2)])

def get_mac_address():
    interfaces = ["wlp3s0", "enp0s25", "enp51s0f0"]
    mac = "00:00:00:00:00:00"
    for interface in interfaces:
        try:
            mac = open('/sys/class/net/' + interface + '/address').readline()
        except:
            pass
    return mac[0:17]


def get_exp_data_path():
    mac_address = get_mac_address()
    path = ""
    assert mac_address in ["2c:60:0c:e4:c0:ee", "b0:68:e6:82:7b:79", "d8:9e:f3:35:97:9b", "3c:ec:ef:38:74:38"], "mac address error"
    if mac_address == "2c:60:0c:e4:c0:ee":  #think pad
        path = None
    elif mac_address == "b0:68:e6:82:7b:79":  # lab computer left
        path = '/media/lenovo/AC74661C7465EA12/experiment_data'
    elif mac_address == "d8:9e:f3:35:97:9b":  # lab computer right
        path = '/media/lenovo/144ED9814ED95C54/experiment_data'
    elif mac_address == "3c:ec:ef:38:74:38":  # lab server
        path = '/home/xiaopeng/experiment_data'
    return path

EXP_DATA_PATH = get_exp_data_path()

if __name__ == "__main__":
    print(get_exp_data_path())
