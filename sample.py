import csv
import functools  # reduce()
import numpy as np


coarselabel_map = {
   # 'null'  : 0,
   'still' : 1,
   'walk'  : 2,
   'run'   : 3,
   'bike'  : 4,
   'car'   : 5,
   'bus'   : 6,
   'train' : 7,
   'subway': 8,
}

# channels corresponding to the columns of <position>_motion.txt files
# ordered according to the SHL dataset documentation.
channels_basic = {
    # [...]
    2: 'Acc_x',
    3: 'Acc_y',
    4: 'Acc_z',
    5: 'Gyr_x',
    6: 'Gyr_y',
    7: 'Gyr_z',
    8: 'Mag_x',
    9: 'Mag_y',
    10: 'Mag_z',
    # 11: 'Ori_w',
    # 12: 'Ori_x',
    # 13: 'Ori_y',
    # 14: 'Ori_z',
    # 15: 'Gra_x',
    # 16: 'Gra_y',
    # 17: 'Gra_z',
    # 18: 'LAcc_x',
    # 19: 'LAcc_y',
    # 20: 'LAcc_z',
    # 21: 'Pressure'
    # [...]
}


def size_of_index(index:list) -> int:
    size = functools.reduce(lambda acc,portion: (portion[1]-portion[0])+acc, index, 0)
    return size


def sample(indexes:dict) -> list:
    """
    Given the index associated to each activity, this function generates
    a single index by selecting portions for each activity. This function
    ensures that:
        - each activity has a sufficient number of representants;
        - the portions have a sufficient number of adjacent segments;
        - the number of segments of each activity is balanced;
    """
    # 1. For each activity:
    #   a. filter portions according to their size, e.g. we do not want portions
    #      containing only 5 segments;
    #   b. compute the average size of the portions;
    min_portion_size = 10
    filtered_indexes = {}
    average_size = {}
    for activity in indexes:
        print('filtering the index of {} ...'.format(activity))
        filtered_indexes[activity] = list(filter(lambda portion: portion[1]-portion[0]>min_portion_size, indexes[activity]))
        average_size[activity] = size_of_index(filtered_indexes[activity])/len(filtered_indexes[activity])
        print(' - average size of {} filtered portions: {}'.format(activity, average_size[activity]))

    # 2. determine how many portions to sample for each activity;
    # 3. sample using np.choice;
    sample = []
    num_segment_per_activity = 2000
    for activity in indexes:
        num_segments = int(num_segment_per_activity/average_size[activity])
        ret = np.random.choice(len(filtered_indexes[activity]), size=num_segments, replace=False)
        print(' - selected portions for {}:{}'.format(activity,ret))
        idx = [filtered_indexes[activity][i] for i in ret]
        print(' - size (# of segments) of this index:{}'.format(size_of_index(idx)))
        for por in idx:
            sample.append(por)

    return sample


def extract_examples(sample_index:list, channel:str) -> None:
    """
    Given a sample index in the form of a list of tuples (start, end)
    specifying the starting and ending point of a portion of examples,
    this function extract the corresponding examples to a new file (e.g.
    Acc_x.txt -> Acc_x_sample.txt)
    """
    # sort list in ascending order
    sample_index_sorted = sorted(sample_index, key=lambda tup: tup[0])

    with open('Torso/'+channel+'.txt', 'r') as input_:
        with open('./'+channel+'_sample.txt', 'w') as output_:
            reader = csv.reader(input_)
            writer = csv.writer(output_)

            try:
                # assume that the indexes are sorted in ascending order
                sample_index_iter = iter(sample_index_sorted)
                idx = next(sample_index_iter)
                for i, row in enumerate(reader):
                    if i > idx[1]:
                        idx = next(sample_index_iter)

                    if idx[0]<i and i<idx[1]:
                        writer.writerow(row)
            except StopIteration:
                pass


def index_of_activity(activity:str) -> list:
    """
    Example:
    ```python
    idxs = activity_index('run')
    print('Portions of activity %s :'.format('run'))
    for index in idxs:
        print('%d: [%d; %d]'.format(index[0][0], index[0][1]))
    ```
    """
    print('Constructing the index of {} ...'.format(activity))
    idxs=[]
    label_of_activity = coarselabel_map[activity]
    with open('Torso/Label.txt') as labels:
        reader = csv.reader(labels, delimiter=' ')

        start = -1
        for i, row in enumerate(reader):
            if (int(row[0]) == label_of_activity) and (start == -1):
                # we found the starting point of a portion
                start = i

            if (int(row[0]) != label_of_activity) and (start != -1):
                # we found the end of a portion
                # print('portion found [{};{}]'.format(start, i-1))
                idxs.append((start, i-1))
                start = -1

    return idxs


def main():
    idxs = {}
    for activity in ['still', 'run']:  # coarselabel_map:
        idxs[activity] = index_of_activity(activity)

    sample_idx = sample(idxs)
    print(sample_idx)

    for _, channel in channels_basic.items():
        extract_examples(sample_idx, channel)


if __name__ == '__main__':
    # activity = 'subway'
    # print('Searching for portions of examples containing {} activity ...'.format(activity))
    # idxs = activity_index(activity)
    # print('Portions of activity {} :'.format(activity))
    # print('portion#id portion#start portion#end size')
    # for i, index in enumerate(idxs):
    #     # print('portion #{}: [{}; {}] (size: {})'.format(i, index[0], index[1], index[1]-index[0]))
    #     print('{} {} {} {}'.format(i, index[0], index[1], index[1]-index[0]))

    # print('Extracting selected portions to a new file ...')
    # extract_examples(idxs)

    main()
