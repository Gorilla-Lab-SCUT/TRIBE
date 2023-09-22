import random
import copy
import torch
import torch.nn.functional as F
import numpy as np
import math


class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0, prediction=-1):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age
        self.prediction = prediction

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age, self.prediction

    def empty(self):
        return self.data == "empty"


class CSTU:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u

        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy

    def per_class_dist(self):
        per_class_occupied = [0] * self.num_class
        for cls, class_list in enumerate(self.data):
            per_class_occupied[cls] = len(class_list)

        return per_class_occupied

    def add_instance(self, instance):
        assert (len(instance) == 3)
        x, prediction, uncertainty = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls, score):
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes()
                return self.remove_from_classes(majority_classes, score)
        else:
            return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes: list[int], score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                uncertainty = item.uncertainty
                age = item.age
                score = self.heuristic_score(age=age, uncertainty=uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls

        if max_class is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            else:
                return False
        else:
            return True

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied == max_occupied:
                classes.append(i)

        return classes

    def heuristic_score(self, age, uncertainty):
        return self.lambda_t * 1 / (1 + math.exp(-age / self.capacity)) + self.lambda_u * uncertainty / math.log(self.num_class)

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return

    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)

        tmp_age = [x / self.capacity for x in tmp_age]

        return tmp_data, tmp_age


class PBRS():

    def __init__(self, capacity, num_class):
        self.num_class = num_class

        self.data = [[[], [], []] for _ in range(self.num_class)] #feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * self.num_class
        self.marker = [''] * self.num_class
        self.capacity = capacity
        pass
    def print_class_dist(self):

        print(self.get_occupancy_per_class())
    def print_real_class_dist(self):

        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] +=1
        print(occupancy_per_class)

    def get_memory(self):

        data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def update_loss(self, loss_list):
        for data_per_cls in self.data:
            feats, cls, dls, _, losses = data_per_cls
            for i in range(len(losses)):
                losses[i] = loss_list.pop(0)

    def add_instance(self, instance):
        assert (len(instance) == 5)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True