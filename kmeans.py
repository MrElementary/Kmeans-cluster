import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import statistics

"""1. We request which file the user would like to use for the input data

   2. We get the desired number of clusters and iterations, I put a cap
   on it just a bit less than the number of colours numpy accepts.
   
   3. We establish the initial centers of the clusters with random
   plot points from the data set.
   
   4. Calculate the euclidian distance from each plot point to each of the cluster centers
   
   5. Then for each user-defined iteration we do the following:
      We calculate which cluster center is the closest to each plot point,
      and assign that cluster number to the plot point.
      Then we calculate the centroid from these clusters and adjust our centers.
      Next, we recalculate the euclidian distance from each plot point to the new cluster centers.
      We assign plot points with new minimum distances to their new clusters.
      And finally we create our plot graph and display.
    
    6. Lastly, we display the requested data from the task papers in the terminal.
"""


# K-Means clustering implementation
# =============================================================================================
# Function to get the data from the CSV file
def read_data_func(filename):
    with open(filename) as file:
        next(file)
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            data_list.append(row)
    return data_list


# =============================================================================================
# Function to determine initial centroids, with random plot points from our data
def initial_centers_func(data, num_of_clusters):
    x_plots = []
    y_plots = []
    for x in data:
        x_plots.append(float(x[1]))
        y_plots.append(float(x[2]))

    random_x_plot = random.sample(x_plots, num_of_clusters)
    random_y_plot = random.sample(y_plots, num_of_clusters)
    random_cluster_vals = [list(x) for x in zip(random_x_plot, random_y_plot)]
    return random_cluster_vals


# =============================================================================================
# Function to calculate the euclidian distance from each plot point to each cluster
def euclid_distance_func(data, clusters):
    temp_dist_list = []
    for line in data:
        for number in clusters:
            x_dist_data = float(line[1])
            y_dist_data = float(line[2])
            x_dist_cluster = number[0]
            y_dist_cluster = number[1]
            x_dist = np.power(x_dist_cluster - x_dist_data, 2)
            y_dist = np.power(y_dist_cluster - y_dist_data, 2)
            combined_dist = x_dist + y_dist
            distance = round(np.sqrt(combined_dist), 2)
            temp_dist_list.append(distance)
    return temp_dist_list


# =============================================================================================
# Function to calculate which centroid is closest to each point and assign that cluster to the point
def min_dist_func():
    combined_list = [eu_list[item:item + cluster_val] for item in range(0, len(eu_list), cluster_val)]
    cluster_number = []

    # Encountered an error where combined_list wasn't creating 392 positions sometimes, if
    # that error occurs this will rewrite.
    while len(combined_list) != len(sample_data):
        temp_val_list = euclid_distance_func(sample_data, random_cluster_plots)
        combined_list = [temp_val_list[item:item + cluster_val] for item in range(0, len(temp_val_list), cluster_val)]
    for val in combined_list:
        closest_cluster = np.argmin(val)
        cluster_number.append(closest_cluster)

    if len(sample_data[0]) == 3:
        for line in sample_data:
            line.append(cluster_number[sample_data.index(line)])
    else:
        for line in sample_data:
            line[-1] = cluster_number[sample_data.index(line)]
    return sample_data


# =============================================================================================
# Function to center the midpoints/centroids for each cluster
def calc_centroids(labeled_data):
    x_plots = []
    y_plots = []
    center_plots = []
    cluster_centers_list = []
    for line in labeled_data:
        x_plots.append(float(line[1]))
        y_plots.append(float(line[2]))
        center_plots.append(line[3])

    for line in set(center_plots):
        temp_list_x = []
        temp_list_y = []
        for val in range(len(x_plots)):
            if center_plots[val] == line:
                temp_list_x.append(x_plots[val])
                temp_list_y.append(y_plots[val])
        x = statistics.mean(temp_list_x), statistics.mean(temp_list_y)
        cluster_centers_list.append(x)

    return cluster_centers_list


# =============================================================================================
# Function to plot the data and display the graph
def plot_data(my_list, centers_list):
    x_plots = []
    y_plots = []
    cluster_num_list = []

    for num in my_list:
        x_plots.append(float(num[1]))
        y_plots.append(float(num[2]))
        cluster_num_list.append(num[-1])

    for num in set(cluster_num_list):
        colour = plot_colours[num]
        for num2 in range(len(x_plots)):
            if cluster_num_list[num2] == num:
                plt.scatter(x_plots[num2], y_plots[num2], c=colour, label=num, alpha=0.4)

    # Creating the plot details
    for num in centers_list:
        plt.scatter(num[0], num[1], marker='x', c='black')
    plt.xlabel("Birth Rate")
    plt.ylabel("Life expectancy")
    plt.yticks(np.arange(0, 100, 5))
    plt.xticks(np.arange(0, 70, 5))
    plt.title(f"Clusters for iteration number {count+1}")
    plt.show()


# ===============================================================================================
# Initialization of the program
# and getting user input for calculations

data_list = []
while True:
    try:
        sample_data = read_data_func(input('''Please enter the file name you want to use:
                                           # data1953.csv
                                           # data2008.csv
                                           # dataBoth.csv\n'''))
        break
    except FileNotFoundError:
        print("Invalid input, please enter again")

while True:
    try:
        cluster_val = int(input("Input cluster amount, up to a max of 8:"))
        if 8 >= cluster_val > 0:
            break
        else:
            print("Invalid amount entered")
    except ValueError:
        print("Invalid input, please enter again")

while True:
    try:
        iteration_val = int(input("Input the number of iterations, up to a max of 6: "))
        if 6 >= iteration_val > 0:
            break
        else:
            print("Invalid amount entered")
    except ValueError:
        print("Invalid input, please enter again")

# Creating initial random cluster centers and measuring the distance to it for
# each plot point
random_cluster_plots = initial_centers_func(sample_data, cluster_val)
eu_list = euclid_distance_func(sample_data, random_cluster_plots)

# Using random colours for clusters each time the program is run
colours = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
plot_colours = random.sample(colours, cluster_val)

# Driver code for the program, utilizing the functions created to center the clusters further
# for each iteration as well as redeploy plot accuracy.
for i in range(iteration_val):
    count = i
    temp_eu_list = min_dist_func()
    cluster_centers = calc_centroids(temp_eu_list)
    eu_list = euclid_distance_func(temp_eu_list, cluster_centers)
    new_data_list = min_dist_func()
    plot_data(new_data_list, cluster_centers)

# =================================================================================================
# Code for display on terminal after graph display

# Lists for use below
cluster_num = []
birthrates = []
life_expectancy = []
country_list = []

# Populating lists with data
for i in new_data_list:
    cluster_num.append(i[-1])
    country_list.append(i[0])
    birthrates.append(float(i[1]))
    life_expectancy.append(float(i[2]))

# Iterating over each cluster and displaying data for that specific cluster
for i in set(cluster_num):
    temp_list_birthrate = []
    temp_list_life = []
    temp_country_list = []
    for j in range(len(birthrates)):
        if cluster_num[j] == i:
            temp_list_birthrate.append(birthrates[j])
            temp_list_life.append(life_expectancy[j])
            temp_country_list.append(country_list[j])
    birth_rate_var = round(statistics.mean(temp_list_birthrate), 2)
    life_expectancy_var = round(statistics.mean(temp_list_life), 2)
    print(
        f"\nThe mean birth rate and life expectancy in cluster {i + 1} is: {birth_rate_var} and {life_expectancy_var}")

    # Number of countries in each cluster
    print(f"The number of countries in cluster {i + 1} is: {len(temp_list_birthrate)}")

    # List of countries for each cluster
    print(f"Countries in cluster {i + 1}:\n{temp_country_list}\n")

# References for sites used
# https://www.tutorialspoint.com/numpy/numpy_insert.htm#:~:text=This%20function%20inserts%20values%20in,the%20input%20array%20is%20flattened.
# https://www.geeksforgeeks.org/numpy-argmin-python/
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# https://docs.python.org/2/library/csv.html
# https://realpython.com/k-means-clustering-python/
# https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
# https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
# https://www.askpython.com/python/examples/k-means-clustering-from-scratch
# Additional reading in the task 22 folder
