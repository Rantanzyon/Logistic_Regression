from load_csv import ft_load_csv
import matplotlib.pyplot as plt

def main():

    try:

        data = ft_load_csv('./datasets/dataset_train.csv')
        features = data.columns[6:]
        houses = ['Gryffindor', 'Hufflepuff', 'Slytherin', 'Ravenclaw']

        for i, feature in enumerate(features):
            print(f'{i}: {feature}')

        print()
        feature = features[int(input('Enter an index of the x-axis feature: '))]

        for house in houses:

            feature_data = data[data['Hogwarts House'] == house][feature].astype(float)
            plt.hist(feature_data, bins=20, density=True, label=house, alpha=0.5)

        plt.title(feature)
        plt.ylabel('Density')
        plt.xlabel(f'{feature} results')
        plt.legend()
        plt.show()

    except BaseException as error:
        print(f'{type(error).__name__}: {error}')

if __name__ == '__main__':
    main()

# def main():

#     try:

#         data = ft_load_csv('./datasets/dataset_train.csv')

#         # fig, axes = plt.subplots(4, 4, figsize=(20,14))
#         # axes = axes.flatten()

#         for i, subject in enumerate(data.columns[6:]):

#             ax = axes[i]
#             ax.set_title(subject)

#             h1 = data[data['Hogwarts House'] == 'Hufflepuff'][subject].dropna()
#             h2 = data[data['Hogwarts House'] == 'Gryffindor'][subject].dropna()
#             h3 = data[data['Hogwarts House'] == 'Ravenclaw'][subject].dropna()
#             h4 = data[data['Hogwarts House'] == 'Slytherin'][subject].dropna()
            
#             ax.hist(h1, bins=20, linewidth=2, density=True, histtype='step', label='Hufflepuff')
#             ax.hist(h2, bins=20, linewidth=2, density=True, histtype='step', label='Gryffindor')
#             ax.hist(h3, bins=20, linewidth=2, density=True, histtype='step', label='Ravenclaw')
#             ax.hist(h4, bins=20, linewidth=2, density=True, histtype='step', label='Slytherin')


#         fig.delaxes(axes[13])
#         fig.delaxes(axes[14])
#         fig.delaxes(axes[15])

#         handles, labels = axes[0].get_legend_handles_labels()
#         fig.legend(handles, labels)
#         plt.subplots_adjust(hspace=0.5)
#         plt.show()

#     except BaseException as error:
#         print(f'{type(error).__name__}: {error}')

# if __name__ == '__main__':
#     main()