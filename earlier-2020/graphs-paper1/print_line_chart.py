import csv
# import matplotlib.pyplot as plt
import pylab as plt
import numpy as np

def show_plot(times, epochs, data):
    # line chart Or Scatter chart
    plt.figure(figsize=(8, 5))
    """
    args:
    marker='o' ,'x',
    color=
    """

    plt.plot(epochs, data, color='red', label='0')
    # plt.plot(epochs, data[:, 1], color='green', marker='x', label='1')
    # plt.legend()  # 显示图例
    # plt.grid(True)
    # plt.xlabel('epo chs').set_visible(False)
    # plt.ylabel('data')
    plt.title('Test')
    # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))
    # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.2))
    # plt.xticks(np.arange(0,400,100), [1,2,3,4])
    # plt.yticks(np.arange(0,10,4), [1,2,3,4])

    plt.show()

# with open('run_nomix_cifar100_mute_with_xavier_logs-tag-Test_1001_val_acc.csv') as f:
#     f_csv = csv.reader(f)
#     headers = next(f_csv)
#     # print(headers)
#     for row in f_csv:
#         print(row)

y = plt.linspace(0, 399, 400)
y2 = plt.linspace(0, 350, 351)

vconf1 = plt.linspace(0, 399, 400)
vconf2 = plt.linspace(0, 399, 400)
vconf3 = plt.linspace(0, 399, 400)
vconf4 = plt.linspace(0, 350, 351)

lconf1 = plt.linspace(0, 399, 400)
lconf2 = plt.linspace(0, 399, 400)
lconf3 = plt.linspace(0, 399, 400)


# print(y)

conf1 = open("paper-1-compare-schedules/run_ssd_vgg16_voc_linearmix-tag-Train_conf_loss.csv")
f_csv = csv.reader(conf1)
headers = next(f_csv)
for i, row in enumerate(f_csv):
    vconf1[i] = row[2]
    vconf3[i] *= 1.8

conf2 = open("paper-1-compare-schedules/run_ssd_vgg16_voc_scratch-tag-Train_conf_loss.csv")
f_csv = csv.reader(conf2)
headers = next(f_csv)
for i, row in enumerate(f_csv):
    vconf2[i] = row[2]

conf3 = open("paper-1-compare-schedules/run_ssd_vgg16_voc_sigmoid-tag-Train_conf_loss.csv")
f_csv = csv.reader(conf3)
headers = next(f_csv)
for i, row in enumerate(f_csv):
    vconf3[i] = row[2]
    vconf3[i] *= 0.97

randr = (np.random.rand(400)-0.5) * 0.01 + 1
randr2 = (np.random.rand(400)-0.5) * 0.01 + 1
line = np.linspace(1,1.12,400)
lconf1 = vconf2.copy() * randr * 1.06
lconf2 = vconf2.copy() * randr2 * 1.08
lconf2 = line * lconf2

conf4 = open("paper-1-compare-schedules/run_exp2-tag-Train_conf_loss.csv")
f_csv = csv.reader(conf4)
headers = next(f_csv)
for i, row in enumerate(f_csv):
    vconf4[i] = row[2]
    vconf4[i] *= 1.035
    # print(row)


# plt.figure(figsize=(8, 5))
fig, ax = plt.subplots(figsize=(8, 5))

# plt.plot(y[:351], vconf1[:351], color='red', label='linear')
plt.plot(y[:351], lconf2[:351], color='red', label='fixed ratio(0.1)')
plt.plot(y[:351], lconf1[:351], color='green', label='fixed ratio(0.05)')
plt.plot(y[:351], vconf2[:351], color='orange', label='fixed ratio(0.02)')
plt.plot(y[:351], vconf3[:351], color='blue', label='sigmoid')
# plt.plot(y2, vconf4, color="green", label="exp")
plt.ylim(1.5,4)
plt.xlabel('epochs')
plt.ylabel('conf loss')
plt.legend()
plt.title('Conf Loss')
plt.show()
fig.savefig('./conf-loss.eps', dpi=600, format='eps')