import random
import os
import numpy as np
import heapq
import copy
import time
import sys


class Map2D:
    def __init__(self, data):
        self.w = len(data[0])  # cols
        self.h = len(data)  # rows
        self.data = np.asarray(data)

    def __getitem__(self, item):
        return self.data[item]

    def replace(self, x, y, a):
        self.data[x, y] = a


class Node:
    def __init__(self, row, col, map2d, action, parent=None):
        self.row = row
        self.col = col
        self.parent = parent
        self.level = map2d
        self.action = action

        if self.parent != None:
            self.step = self.parent.step + 1
        else:
            self.step = 0

        self.score = -1

    def get_key(self):
        result = ""
        for x in range(self.level.h):
            for y in range(self.level.w):
                result += str(self.level[x][y])
        return "{},{},{}".format(result, self.row, self.col)

    def get_score(self, goal_row, goal_col):
        # manhattan distance to goal
        if self.score == -1:
            self.score = abs(self.row - goal_row) + abs(self.col - goal_col) + self.step
        return self.score

    # returns next valid actions
    def get_actions(self):
        level = self.level
        left_end = 0
        right_end = self.level.w - 1
        top = 0
        bottom = self.level.h - 1
        row = self.row
        col = self.col
        actions = []
        # if current position is ladder
        if level[row, col] == '#':
            # if player is on the lowest row
            if row == bottom:
                if col != left_end and level[row, col - 1] != 'b' and level[row, col - 1] != 'B': actions.append("left")
                if col != right_end and level[row, col + 1] != 'b' and level[row, col + 1] != 'B': actions.append(
                    "right")
                if row != top and level[row - 1, col] != 'b' and level[row - 1, col] != 'B': actions.append("up")
            # if player is not on the lowest row
            elif row != bottom:
                if row != top and level[row - 1, col] != 'b' and level[row - 1, col] != 'B': actions.append("up")
                if level[row + 1, col] != 'b' and level[row + 1, col] != 'B': actions.append("down")
                if col != left_end and (level[row, col - 1] == '#' or level[row, col - 1] == '-'): actions.append(
                    "left")
                if col != left_end and (
                        level[row, col - 1] == 'G' or level[row, col - 1] == 'E' or level[row, col - 1] == '.') and (
                        level[row + 1, col - 1] == 'b' or level[row + 1, col - 1] == 'B' or level[
                    row + 1, col - 1] == '#'): actions.append("left")
                if col != right_end and (level[row, col + 1] == '#' or level[row, col + 1] == '-'): actions.append(
                    "right")
                if col != right_end and (
                        level[row, col + 1] == 'G' or level[row, col + 1] == 'E' or level[row, col + 1] == '.') and (
                        level[row + 1, col + 1] == 'b' or level[row + 1, col + 1] == 'B' or level[
                    row + 1, col + 1] == '#'): actions.append("right")
                if col != left_end and (
                        level[row, col - 1] == 'G' or level[row, col - 1] == 'E' or level[row, col - 1] == '.') and (
                        level[row + 1, col - 1] != 'b' and level[row + 1, col - 1] != 'B' and level[
                    row + 1, col - 1] != '#'): actions.append("d-left")
                if col != right_end and (
                        level[row, col + 1] == 'G' or level[row, col + 1] == 'E' or level[row, col + 1] == '.') and (
                        level[row + 1, col + 1] != 'b' and level[row + 1, col + 1] != 'B' and level[
                    row + 1, col + 1] != '#'): actions.append("d-right")

        # if current position is rope
        elif level[row, col] == '-':
            # if player is on the lowest row
            if row == bottom:
                if col != left_end and level[row, col - 1] != 'b' and level[row, col - 1] != 'B': actions.append("left")
                if col != right_end and level[row, col + 1] != 'b' and level[row, col + 1] != 'B': actions.append(
                    "right")
            # if player is not on the lowest row
            elif row != bottom:
                if level[row + 1, col] != 'b' and level[row + 1, col] != 'B': actions.append("down")
                if col != left_end and (level[row, col - 1] == '#' or level[row, col - 1] == '-'): actions.append(
                    "left")
                if col != left_end and (
                        level[row, col - 1] == 'G' or level[row, col - 1] == 'E' or level[row, col - 1] == '.') and (
                        level[row + 1, col - 1] == 'b' or level[row + 1, col - 1] == 'B' or level[
                    row + 1, col - 1] == '#'): actions.append("left")
                if col != right_end and (level[row, col + 1] == '#' or level[row, col + 1] == '-'): actions.append(
                    "right")
                if col != right_end and (
                        level[row, col + 1] == 'G' or level[row, col + 1] == 'E' or level[row, col + 1] == '.') and (
                        level[row + 1, col + 1] == 'b' or level[row + 1, col + 1] == 'B' or level[
                    row + 1, col + 1] == '#'): actions.append("right")
                if col != left_end and (
                        level[row, col - 1] == 'G' or level[row, col - 1] == 'E' or level[row, col - 1] == '.') and (
                        level[row + 1, col - 1] != 'b' and level[row + 1, col - 1] != 'B' and level[
                    row + 1, col - 1] != '#'): actions.append("d-left")
                if col != right_end and (
                        level[row, col + 1] == 'G' or level[row, col + 1] == 'E' or level[row, col + 1] == '.') and (
                        level[row + 1, col + 1] != 'b' and level[row + 1, col + 1] != 'B' and level[
                    row + 1, col + 1] != '#'): actions.append("d-right")

        # if current position is empty or gold or enemy
        elif level[row, col] == '.' or level[row, col] == 'G' or level[row, col] == 'E':
            # if player is not on the lowest row
            if row != bottom:
                # below is empty or rope or gold
                if level[row + 1, col] != 'b' and level[row + 1, col] != 'B' and level[row + 1, col] != '#':
                    actions.append("down")

                # below is block or ladder
                elif level[row + 1, col] == 'b' or level[row + 1, col] == 'B' or level[row + 1, col] == '#':
                    if col != left_end and (level[row, col - 1] == '#' or level[row, col - 1] == '-'): actions.append(
                        "left")
                    if col != left_end and (level[row, col - 1] == 'G' or level[row, col - 1] == 'E' or level[
                        row, col - 1] == '.') and (
                            level[row + 1, col - 1] == 'b' or level[row + 1, col - 1] == 'B' or level[
                        row + 1, col - 1] == '#'): actions.append("left")
                    if col != right_end and (level[row, col + 1] == '#' or level[row, col + 1] == '-'): actions.append(
                        "right")
                    if col != right_end and (level[row, col + 1] == 'G' or level[row, col + 1] == 'E' or level[
                        row, col + 1] == '.') and (
                            level[row + 1, col + 1] == 'b' or level[row + 1, col + 1] == 'B' or level[
                        row + 1, col + 1] == '#'): actions.append("right")
                    if level[row + 1, col] == '#': actions.append("down")
                    if col != left_end and (level[row, col - 1] == 'G' or level[row, col - 1] == 'E' or level[
                        row, col - 1] == '.') and (
                            level[row + 1, col - 1] != 'b' and level[row + 1, col - 1] != 'B' and level[
                        row + 1, col - 1] != '#'): actions.append("d-left")
                    if col != right_end and (level[row, col + 1] == 'G' or level[row, col + 1] == 'E' or level[
                        row, col + 1] == '.') and (
                            level[row + 1, col + 1] != 'b' and level[row + 1, col + 1] != 'B' and level[
                        row + 1, col + 1] != '#'): actions.append("d-right")


            # if player is on the lowest row
            elif row == bottom:
                if col != left_end and level[row, col - 1] != 'b' and level[row, col - 1] != 'B': actions.append("left")
                if col != right_end and level[row, col + 1] != 'b' and level[row, col + 1] != 'B': actions.append(
                    "right")

        # print("{},{} actions : {}".format(row,col,actions))
        return actions

        # returns children of a node

    def get_children(self):
        child_nodes = []
        # get next possible actions
        valid_actions = self.get_actions()
        # for each action create a child node
        for i in range(len(valid_actions)):
            action = valid_actions[i]
            # next position based on action
            if action == 'left':
                child_row = self.row
                child_col = self.col - 1
            elif action == 'right':
                child_row = self.row
                child_col = self.col + 1
            elif action == 'up':
                child_row = self.row - 1
                child_col = self.col
            elif action == 'down':
                child_row = self.row + 1
                child_col = self.col
            elif action == 'd-left':
                child_row = self.row + 1
                child_col = self.col - 1
            elif action == 'd-right':
                child_row = self.row + 1
                child_col = self.col + 1
            else:
                print("error")

                # create child node
            child_level = self.level
            child = Node(child_row, child_col, child_level, action, self)
            child_nodes.append(child)

        # print(len(valid_actions), len(child_nodes))
        return child_nodes

    def get_path(self):
        path = list()
        other_golds = list()
        node = self
        path.append((node.row, node.col))
        while node.parent != None:
            path.append((node.parent.row, node.parent.col))
            if self.level[node.parent.row, node.parent.col] == 'G':
                other_golds.append((node.parent.row, node.parent.col))
            node = node.parent
        return path, other_golds


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority, priority2):
        heapq.heappush(self.elements, (priority, priority2, item))

    def get(self):
        return heapq.heappop(self.elements)[2]

    def display(self):
        for i in range(len(self.elements)):
            print(self.elements[i][2].row, self.elements[i][2].col, self.elements[i][2].score, self.elements[i][0],
                  self.elements[i][1])


class AStar():
    def __init__(self, root, goal_x, goal_y):
        self.root = root
        self.goal_row = goal_x
        self.goal_col = goal_y

    def run(self, timer):
        cnt = 0
        score = self.root.get_score(self.goal_row, self.goal_col)
        queue = PriorityQueue()
        # push node in the list
        queue.put(self.root, score, cnt)
        visited = set()

        # while the list is not empty
        while not queue.empty():
            if time.time() - timer > 1:
                return None
            current = queue.get()
            if (current.get_key() not in visited):
                # check for goal condition
                if current.row == self.goal_row and current.col == self.goal_col:
                    # print('goal')
                    return current
                visited.add(current.get_key())
                children = current.get_children()
                for c in (children):
                    cnt += 1
                    score = c.get_score(self.goal_row, self.goal_col)
                    # print(score)
                    queue.put(c, score, cnt)
        return None


def count_elements(level):
    golds = list()
    for i in range(level.h):
        for j in range(level.w):
            if level[i, j] == 'G':
                golds.append((i, j))
    return golds


def find_all_golds(root, golds, map2d):
    timer = time.time()
    total_dist = 0
    gold_found = list()
    dig = False
    for g in golds:
        if time.time() - timer > 1:
            return len(gold_found), total_dist
        if g not in gold_found:
            astar = AStar(root, g[0], g[1])
            to_goal = astar.run(timer)
            if to_goal != None:
                if time.time() - timer > 1:
                    return len(gold_found), total_dist
                root2 = Node(g[0], g[1], to_goal.level, None, None)
                astar = AStar(root2, root.row, root.col)
                to_start = astar.run(timer)
                if to_start != None:
                    # print('goal')
                    gold_found.append((g[0], g[1]))
                    path, other_golds = to_goal.get_path()
                    total_dist += len(path)
                    for og in other_golds:
                        if og not in gold_found and og in golds: gold_found.append(og)
                    path, other_golds = to_start.get_path()
                    for og in other_golds:
                        if og not in gold_found and og in golds: gold_found.append(og)
    return len(gold_found), total_dist


def get_level(file_name):
    sample = []
    with open(file_name, 'r') as current_file:
        for line in current_file.readlines():
            line = list(line.rstrip('\n'))
            transformed_line = []
            for item in line:
                transformed_line.append(item)
            sample.append(transformed_line)
    return sample


def get_starting_point(map2d):
    golds = []
    row = 0
    col = 0
    for i in range(map2d.h):
        for j in range(map2d.w):
            if map2d[i, j] == 'M':
                row = i
                col = j
                break
    while row != map2d.h - 1 and map2d[row + 1, col] != 'B' and map2d[row + 1, col] != 'b' and map2d[
        row + 1, col] != '#' and map2d[row, col] != '-':
        row = row + 1
        if map2d[row, col] == 'G':
            golds.append((row, col))
    return row, col, golds


def get_hamm_dist(golds, row, col):
    # manhattan distance to golds
    total_dist = 0
    for g in golds:
        dist = abs(row - g[0]) + abs(col - g[1])
        total_dist += dist
    return total_dist


def get_gold_dist(golds):
    cnt = 0
    total_dist = 0
    avg_dist = 0
    if len(golds) > 1:
        for i in range(len(golds)):
            for j in range(i + 1, len(golds)):
                cnt += 1
                g1 = golds[i]
                g2 = golds[j]
                dist = abs(g1[0] - g2[0]) + abs(g1[1] - g2[1])
                total_dist += dist

        avg_dist = total_dist / cnt
    return avg_dist

from pdb import set_trace as TT

def get_score(level):
    timer = time.time()
    map2d = Map2D(level)
    all_golds = count_elements(map2d)
    row, col, coll_on_start = get_starting_point(map2d)
    golds = [g for g in all_golds if g not in coll_on_start]
    map2d.replace(row, col, '.')

    score = 0
    dist = 0
    path_len = 0

    if len(all_golds) == 0:
        score = -1
    else:
        root = Node(row, col, map2d, None, None)
        seq, path_len = find_all_golds(root, golds, map2d)
        collected = seq + len(coll_on_start)
        score = 1 / (1 + (len(all_golds) - collected))
        # dist = get_gold_dist(all_golds)
    # print(time.time() - timer)
    return score, path_len
