import os
from typing import List

class Logger():

    def __init__(self, file_path, logger_type='csv', columns=[], overwrite=False, index_col=0) -> None:
        
        self.file_path = file_path
        self.logger_type = logger_type
        assert self.logger_type in ['csv']

        self.columns = columns
        self.index_col = index_col # column to determine if a row is already done
        self.done_index = set()

        self.initiate(file_path, columns, overwrite)
    
    def make_row(self, data:List):
        if self.logger_type == 'csv':
            return ",".join([str(d) for d in data]) + '\n'
        else:
            raise ValueError('not implemented yet')
    
    def split_row(self, row):
        if self.logger_type == 'csv':
            return row.split(',')
        else:
            raise ValueError('not implemented yet')

    def initiate(self, file_path, columns, overwrite):
        if not os.path.isfile(file_path) or overwrite:
            # no previous file exists or overwrite is on. Then create a new file
            with open(file_path, 'w') as f:
                f.write(self.make_row(columns))
        else:
            self.done_index, col_name = self.make_done_index(self.index_col)
            print(f'loaded {len(self.done_index)} unique index from donelist. \n Index column name: {col_name}')
    
    def get_column(self, col_index):
        if not os.path.isfile(self.file_path):
            return []
        with open(self.file_path, 'r') as f:
            data = f.readlines()
            # first row is a column name
            column = [self.split_row(row)[col_index] for row in data[1:]]
            col_name = self.split_row(data[0])[col_index]
            return column, col_name

    def make_done_index(self, index_col):
        done_list, col_name = self.get_column(index_col)
        return set(done_list), col_name

    def add_row(self, data:List):
        assert len(data) == len(self.columns), 'row data lenght is not same as column name length'
        
        with open(self.file_path, 'a') as f:
            row = self.make_row(data)
            f.write(row)
            self.done_index.add(data[self.index_col])

    def check_done(self, index):
        if index in self.done_index:
            return True
        return False

    

def num_param(model):
    return sum(p.numel() for p in model.parameters()) / 1000000


def colored_list(lst):
    from colorama import init
    from colorama import Fore, Back, Style
    init()
    max_lst = max(lst)
    min_lst = min(lst)
    items = []
    for i in lst:
        if i == max_lst:
            items.append(Fore.RED + f"{i:.2f}" + Style.RESET_ALL)
        elif i == min_lst:
            items.append(Fore.BLUE + f"{i:.2f}" + Style.RESET_ALL)
        else:
            items.append(f"{i:.2f}")
    return " ".join(items)