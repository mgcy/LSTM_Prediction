# Try to predict label_1
train_len = 30
overlap = 8
f1 = open('D:\Academic\lstm_prediction\\test_data.txt', 'w')
label1 = [str(l.split(',')[2]) for l in open('D:\Academic\data_nvspl\LAKE017\\NVSPL_LAKE017_2011_03_09_07.txt', 'r')]
for i1 in range(2155 - train_len, 2155):
    # write upper part data
    f1.write(str(label1[i1]))
    f1.write('\n')

for i2 in range(2155, 2162):
    # write overlapped data
    f1.write(str(label1[i2]))
    f1.write('\n')

for i3 in range(2162, 2162 + train_len+1):
    # write lower part data
    f1.write(str(label1[i3]))
    f1.write('\n')
f1.close()
f4 = open('D:\Academic\lstm_prediction\\test_data.txt', 'r')
label_full = [str(l.split(' ')[0]) for l in f4]

f2 = open('D:\Academic\lstm_prediction\\test_data_upperpart_test.txt', 'w')
for i4 in range(train_len + overlap + 1 - 1):
    f2.write(str(label_full[i4]))
f2.close()

f3 = open('D:\Academic\lstm_prediction\\test_data_lowerpart_test.txt', 'w')
for i5 in range(len(label_full)-train_len - overlap, len(label_full)):
    f3.write(str(label_full[i5]))
f3.close()

f4.close()
