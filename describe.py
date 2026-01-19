from load_csv import ft_load_csv
import pandas as pd

class statistics:

    @classmethod
    def count(self, data: list) -> int:
        length = 0
        for _ in data:
            length += 1
        return length
    
    @classmethod
    def mean(self, data: list) -> float:
        length = self.count(data)
        total = 0
        for e in data:
            total += e
        return total / length
    
    @classmethod
    def var(self, data: list) -> float:
        mean = self.mean(data)
        length = self.count(data)
        total = 0
        for e in data:
            total += (e - mean) ** 2
        return total / length
    
    @classmethod
    def std(self, data: list) -> float:
        return self.var(data) ** 0.5
    
    @classmethod
    def min(self, data: list) -> float:
        data.sort()
        return data[0]
    
    @classmethod
    def max(self, data: list) -> float:
        data.sort(reverse=True)
        return data[0]

    @classmethod
    def median(self, data: list) -> float:
        data.sort()
        length = self.count(data)
        if length % 2:
            return data[length // 2]
        else:
            return self.mean([data[length // 2], data[length // 2 - 1]])

    @classmethod
    def q1(self, data: list) -> float:
        data.sort()
        length = self.count(data)
        mid_index = length // 2
        return self.median(data[:mid_index + (1 if length % 2 else 0)])
    
    @classmethod
    def q3(self, data: list) -> float:
        data.sort()
        length = self.count(data)
        mid_index = length // 2
        return self.median(data[mid_index:])
    
    @classmethod
    def all_stats(self, data: pd.DataFrame, subject: str) -> dict:
        arr = data[subject].astype(float)
        arr = [e for e in arr if e == e]
        return {
            'Count': self.count(arr),
            'Mean': self.mean(arr),
            'Std': self.std(arr),
            'Min': self.min(arr),
            '25%': self.q1(arr),
            '50%': self.median(arr),
            '75%': self.q3(arr),
            'Max': self.max(arr)
        }
    
def display_block(*args):

    subjects = args[0]
    names = args[1:]

    text_name = ''.ljust(7)
    text_count = 'Count'.ljust(7)
    text_mean = 'Mean'.ljust(7)
    text_std = 'Std'.ljust(7)
    text_min = 'Min'.ljust(7)
    text_q1 = '25%'.ljust(7)
    text_median = '50%'.ljust(7)
    text_q3 = '75%'.ljust(7)
    text_max = 'Max'.ljust(7)

    for name in names:
        width = 20
        subname = name
        if len(name) > 12:
            subname = name[:12] + '...'

        text_name += subname.rjust(width)
        text_count += f'{subjects[name]['Count']:.6f}'.rjust(width)
        text_mean += f'{subjects[name]['Mean']:.6f}'.rjust(width)
        text_std += f'{subjects[name]['Std']:.6f}'.rjust(width)
        text_min += f'{subjects[name]['Min']:.6f}'.rjust(width)
        text_q1 += f'{subjects[name]['25%']:.6f}'.rjust(width)
        text_median += f'{subjects[name]['50%']:.6f}'.rjust(width)
        text_q3 += f'{subjects[name]['75%']:.6f}'.rjust(width)
        text_max += f'{subjects[name]['Max']:.6f}'.rjust(width)

    print(text_name)
    print(text_count)
    print(text_mean)
    print(text_std)
    print(text_min)
    print(text_q1)
    print(text_median)
    print(text_q3)
    print(text_max)


def main():

    try:

        data = ft_load_csv('./datasets/dataset_train.csv')
        subjects = {subject: statistics.all_stats(data, subject) for subject in data.columns[6:]}
        
        print()
        display_block(subjects, 'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts')
        print()
        display_block(subjects, 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic')
        print()
        display_block(subjects, 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying')

    except BaseException as error:
        print(f'{type(error).__name__}: {error}')

if __name__ == '__main__':
    main()