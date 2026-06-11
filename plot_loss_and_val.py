import json
import matplotlib.pyplot as plt

filepath = './output/DCNN/log.txt'
train_loss_list = []
test_loss_list = []
val_list = []
epoch_list = []
with open(filepath, 'r') as f:
    lines = f.readlines()
    e = 0
    for line in lines:
        json_data = json.loads(line)
        train_loss_list.append(json_data['train_loss_0'])
        test_loss_list.append(json_data['test_loss'])
        val_list.append(json_data['test_acc1'])
        epoch_list.append(e)
        e += 1
    f.close()


config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 12
}
plt.rcParams.update(config)

plt.figure()
plt.plot(epoch_list, train_loss_list, 'b', label='Training Loss')
plt.plot(epoch_list, test_loss_list, 'r', label='Testing Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and Testing Loss')
plt.legend(loc='upper right')
plt.savefig('loss.png', format='png')

plt.figure()
plt.plot(epoch_list, val_list, 'b', label='Testing ACC')
plt.title('Testing Accurate')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(loc='lower right')
plt.savefig('acc.png', format='png')