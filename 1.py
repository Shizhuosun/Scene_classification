
mapping_n2o = {"Coast": 0, "Forest": 1, "Highway": 2, "Insidecity": 3, "Mountain": 4, "Office": 5, "OpenCountry": 6, "Street": 7, "Suburb": 8, "TallBuilding": 9,
                                "bedroom": 10, "industrial": 11, "kitchen": 12, "livingroom": 13,"store": 14}

pairs = []
with open('clas.txt', 'r', encoding='utf-8') as f:
    for line in f:
        ld = line.strip()
        if ld:
            ll = ld.split(':')
            lt = tuple([ll[1], ll[0]])
            pairs.append(lt)
print('Num test samples:', len(pairs))
print(pairs)