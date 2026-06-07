import csv
import json
import gzip

concepts = set()

with gzip.open("data/conceptnet-assertions-5.7.0.csv.gz", "rt", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        try:
            # column 2 and 3 are the two concept nodes
            for col in [2, 3]:
                parts = row[col].split("/")
                if len(parts) > 3 and parts[2] == "en":
                    concepts.add(parts[3].replace("_", " "))
            if len(concepts) >= 50000:
                break
        except:
            continue

concepts = list(concepts)[:50000]  # cap at 50k to start
with open("data/concepts.json", "w") as f:
    json.dump(concepts, f)

print(f"saved {len(concepts)} concepts")
