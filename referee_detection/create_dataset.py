"""
test for getting a dataset and tracking it with dvc
"""
import csv

with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["name", "age", "country"]

    writer.writerow(field)