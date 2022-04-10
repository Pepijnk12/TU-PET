import json

with open("data/mfd2.0.dic", "r") as f:
    f.readline()
    value_dict = {}
    rename_dict = {
        "care.virtue": "care",
        "care.vice": "harm",
        "fairness.virtue": "fairness",
        "fairness.vice": "cheating",
        "loyalty.virtue": "loyalty",
        "loyalty.vice": "betrayal",
        "authority.virtue": "authority",
        "authority.vice": "subversion",
        "sanctity.virtue": "purity",
        "sanctity.vice": "degradation"
    }

    for index in range(10):
        value = f.readline()
        value = value.replace("\n", "")
        value = value.split("\t")
        value_dict[value[0]] = rename_dict[value[1]]

    big_verbalizer = {}
    for value in value_dict.values():
        big_verbalizer[value] = []

    f.readline()
    data = f.readlines()
    for element in data:
        clean_element = element.replace("\n", "").split("\t")
        moral_value = clean_element[0]
        moral_id = clean_element[1]
        big_verbalizer[value_dict[moral_id]].append(moral_value)


    # Find duplicate labels
    s = set()
    remove_labels = []
    for k, v in big_verbalizer.items():
        for x in v:
            if x in s:
                remove_labels.append(x)
            s.add(x)
    print("Len duplicates", len(remove_labels))

    # Remove duplicate labels
    for x in remove_labels:
        for k, v in big_verbalizer.items():
            if x in v:
                v.remove(x)

    for k, v in big_verbalizer.items():
        print(k, v)
        print(k, len(v))

    with open("data/big_verbalizer.json", "w+") as f:
       json.dump(big_verbalizer, f)
