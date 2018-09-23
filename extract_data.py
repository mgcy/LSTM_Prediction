# Copyright 2018 Yifan Yang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
'''
Usage: This code generates data file for LSTM prediction. So far, only label 1.1
        was selected to train and test the LSTM model. The data was separated into
        upper part and lower part.
Parameters:
    date_days: the days of data
    train_len: the length of data above and below the overlaps.
    output folder: D:\Academic\lstm_prediction\overlap_data
    f2: Some information about overlaps with format:
        [date hr start_time end_time len_overlap label1 label2]
'''

date_time = [str(l.split()[0]) for l in open("D:\Academic\data_nvspl\SRCID_LAKE017.txt")]

f2 = open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_info.txt", 'w')

# Make the date from the first three days
date_days = 16
train_len = 30

# Get the number of days
date_index = []
[date_index.append(i) for i in date_time if not i in date_index]
len_date = len(date_index)
print('The max number of days: ' + str(len_date))

for i1 in range(date_days):
    # Choose a day
    day_index = date_index[i1]
    hr_index = []
    hr = []
    # Get the hour information first
    for l in open("D:\Academic\data_nvspl\SRCID_LAKE017.txt"):
        if l.split()[0] == day_index:
            hr_index.append(l.split()[1])
            [hr.append(i) for i in hr_index if not i in hr]

    # Write the hourly data in the temporary file f1
    for i3 in range(24):
        f1 = open("D:\Academic\data_nvspl\SRCID_LAKE017_expand_labels.txt", 'w')

        for l in open("D:\Academic\data_nvspl\SRCID_LAKE017.txt"):

            if l.split()[0] == day_index:
                if str(i3) in hr:
                    if l.split()[1] == str(i3):
                        f1.write(l)
        f1.close()
        # sort lines in terms of hours and seconds
        lines = open("D:\Academic\data_nvspl\SRCID_LAKE017_expand_labels.txt", 'r').readlines()
        f3 = open("D:\Academic\data_nvspl\SRCID_LAKE017_expand_labels_sort.txt", 'w')
        for line in sorted(lines, key=lambda line: int(line.split()[2])):
            f3.write(line)
        f3.close()

        # Read f1 and write labels in f2. It is possible that during some hours no event occurs.
        if str(i3) in hr:

            start_time = [str(l.split()[2]) for l in
                          open("D:\Academic\data_nvspl\SRCID_LAKE017_expand_labels_sort.txt")]
            len_time = [str(l.split()[3]) for l in
                        open("D:\Academic\data_nvspl\SRCID_LAKE017_expand_labels_sort.txt")]
            labels = [str(l.split()[4]) for l in
                      open("D:\Academic\data_nvspl\SRCID_LAKE017_expand_labels_sort.txt")]
            for i4 in range(len(start_time) - 1):
                if int(start_time[i4 + 1]) < (int(start_time[i4]) + int(len_time[i4])):
                    # if overlap happens
                    end_time = int(start_time[i4]) + int(len_time[i4])
                    len_overlap = int(end_time - 1) - int(start_time[i4 + 1]) + 1

                    # delete len_overlap less than or equal to 3 and # delete label = 1
                    if len_overlap >= 3 and str(labels[i4]) != '1' and str(labels[i4 + 1]) != '1':
                        f2.write(day_index + ' ')
                        f2.write(str(i3) + ' ')
                        f2.write(str(start_time[i4 + 1]) + ' ' + str(end_time - 1) + ' ')
                        f2.write(str(len_overlap) + ' ')
                        f2.write(str(labels[i4]) + ' ' + str(labels[i4 + 1]))
                        f2.write('\n')
f2.close()

date_overlap = [str(l.split()[0]) for l in
                open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_info.txt")]
hr_overlap = [int(l.split()[1]) for l in
              open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_info.txt")]
overlap_start = [int(l.split()[2]) for l in
                 open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_info.txt")]
overlap_end = [int(l.split()[3]) for l in
               open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_info.txt")]
len_overlap_full = [int(l.split()[4]) for l in
                    open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_info.txt")]
up_label = [float(l.split()[5]) for l in
            open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_info.txt")]
low_label = [float(l.split()[6]) for l in
             open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_info.txt")]

# train upper and lower 1.1 up to now.
len_full = len(date_overlap)
len_up = up_label.count(1.1)  # 529
len_lower = low_label.count(1.1)  # 511

i6 = 1
i8 = 1
for i5 in range(len_full):
    # write upper part with label 1.1 data, i.e., up and i5=1.1
    if up_label[i5] == 1.1:
        # check whether overflow
        if (overlap_start[i5] - train_len) >= 1:
            # generate dir in terms of data and hr
            if hr_overlap[i5] < 10:
                data_dir = 'D:\Academic\data_nvspl\LAKE017\\NVSPL_LAKE017_' + str(
                    date_overlap[i5].replace('-', '_')) + '_0' + str(hr_overlap[i5]) + '.txt'

            else:
                data_dir = 'D:\Academic\data_nvspl\LAKE017\\NVSPL_LAKE017_' + str(
                    date_overlap[i5].replace('-', '_')) + '_' + str(hr_overlap[i5]) + '.txt'
            print(data_dir)
            tmpdir = 'D:\\Academic\\lstm_prediction\\overlap_data\\upper\\' + str(i6) + '.txt'
            i6 = i6 + 1
            print(tmpdir)
            tmpdata = [str(l.split(',')[2]) for l in open(data_dir, 'r')]
            f4 = open(tmpdir, 'w')
            # if data with overlap is wanted, use first loop!
            # for i7 in range(overlap_start[i5] - train_len, overlap_end[i5] + 1):
            for i7 in range(overlap_start[i5] - train_len, overlap_start[i5]):
                f4.write(str(tmpdata[i7]) + '\n')
            f4.close()

    #  write lower part with label 1.1 data
    if low_label[i5] == 1.1:
        # check whether overflow
        if (overlap_end[i5] + train_len + 1) <= 3600:
            # generate dir in terms of data and hr
            if hr_overlap[i5] < 10:
                data_dir = 'D:\Academic\data_nvspl\LAKE017\\NVSPL_LAKE017_' + str(
                    date_overlap[i5].replace('-', '_')) + '_0' + str(hr_overlap[i5]) + '.txt'

            else:
                data_dir = 'D:\Academic\data_nvspl\LAKE017\\NVSPL_LAKE017_' + str(
                    date_overlap[i5].replace('-', '_')) + '_' + str(hr_overlap[i5]) + '.txt'
            print(data_dir)
            tmpdir = 'D:\\Academic\\lstm_prediction\\overlap_data\\lower\\' + str(i8) + '.txt'
            i8 = i8 + 1
            print(tmpdir)
            tmpdata = [str(l.split(',')[2]) for l in open(data_dir, 'r')]
            f4 = open(tmpdir, 'w')
            # if data with overlap is wanted, use first loop!
            # for i7 in range(overlap_start[i5], overlap_end[i5] + train_len + 1):
            for i7 in range(overlap_end[i5] + 1, overlap_end[i5] + train_len + 1):
                f4.write(str(tmpdata[i7]) + '\n')
            f4.close()
