import configparser # python 3.x
# import ConfigParser # python 2.x

config = configparser.ConfigParser()

# 用法1
config["DEFAULT"] = {
    "Country": "China",
    "Max_Iter": "2",
}

# 用法2
config["section_1"] = {}
config["section_1"]["a"] = "2"

# 用法3
config["section_2"] = {}
section2 = config["section_2"]
section2["b"] = "2"

config['DEFAULT']['ForwardX11'] = 'yes'

with open("example.ini", "w") as fp:
    config.write(fp)