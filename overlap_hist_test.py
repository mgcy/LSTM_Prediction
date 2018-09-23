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
Usage: Count the information of the overlaps between different signals.

Parameters:
    f2: the target output file
    date_days: how many days wanted to be included
'''
import collections
import matplotlib.pyplot as plt
import statistics

date_time = [str(l.split()[0]) for l in open("D:\Academic\data_nvspl\SRCID_LAKE017.txt")]

# Aim label file is f2, while f1 is a temporary file used for hourly data

f2 = open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_hist_labels.txt", 'w')

# Make the date from the first three days
date_days = 16

# Get the number of days
date_index = []
[date_index.append(i) for i in date_time if not i in date_index]
len_date = len(date_index)
print('The max number of days: ' + str(len_date))
len_overlap = []

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
                    len_overlap.append(end_time - int(start_time[i4 + 1]))
                    for i5 in range(int(start_time[i4 + 1]), end_time):
                        # write date
                        f2.write(day_index + ' ')
                        # write hour
                        f2.write(str(i3) + ' ')
                        f2.write(str(i5) + ' ' + str(labels[i4]) + ' ' + str(labels[i4 + 1]))
                        f2.write('\n')
f2.close()

# print some information of overlap
avg_len_overlap = statistics.mean(len_overlap)
print('The average duration of overlaps')
print(avg_len_overlap)
sum_overlap = sum(len_overlap[0:len(len_overlap)])
print('The sum of overlaps and the proportion:')
print(sum_overlap)
print(str(sum_overlap/13824)+'%')
plt.figure(0)
plt.hist(len_overlap, 30, alpha=0.5, ec='black')
plt.title('Duration Histogram')
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.axvline(avg_len_overlap, color='r', linestyle='dashed', linewidth=2)
print('Counter information of duations: ')
print(collections.Counter(len_overlap))


labels_1 = [str(l.split()[3]) for l in open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_hist_labels.txt")]
labels_2 = [str(l.split()[4]) for l in open("D:\Academic\data_nvspl\SRCID_LAKE017_overlap_hist_labels.txt")]
print('Counter information of first labels: ')
print(collections.Counter(labels_1))
print('Counter information of second labels: ')
print(collections.Counter(labels_2))
