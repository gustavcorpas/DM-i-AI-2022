from concurrent.futures import process
import pandas as pd

data = pd.read_csv('./data/CompleteDataset.csv', sep=',', on_bad_lines="skip")

length_before = len(data["review"])

processed = []

for index, el in enumerate(data["review"]):
    rev = str(el)
    if len(rev) < 10:
        print("removing: " + rev)
        continue
    
    processed.append([data["rating"][index], rev])

df = pd.DataFrame(processed, columns =['rating', 'review'], dtype = int) 

df.to_csv('./data/CompleteDatasetCleaned.csv', index=False)

print(df) 

print(".")
print(".....DONE.....")
print(".")
print("Started at items: " + str(length_before))
print("Ended with items: " + str(len(df["rating"])) + " / " + str(len(df["review"])))
print("Removed items: " + str(length_before - len(df["rating"])))
print(".")


